import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class DummyClusterer:
    def __init__(self, *args, **kwargs):
        pass

    def fit_predict(self, vectors):
        n = len(vectors)
        labels = np.zeros(n, dtype=int)
        if n:
            labels[n // 2 :] = 1
        return labels


@pytest.fixture
def cluster_client(monkeypatch):
    main.AUTH_KEY = "test-key"
    
    # Mock HDBSCAN to avoid actual clustering
    monkeypatch.setattr(main.hdbscan, "HDBSCAN", DummyClusterer)

    # Mock the method on the class, not the module function
    def identity_reduce(self, vectors):
        return vectors

    monkeypatch.setattr(main.VideoClusterer, "_reduce_and_normalize", identity_reduce)
    
    # Also need to mock KNeighborsClassifier since we use it now
    class DummyKNN:
        def __init__(self, n_neighbors=1):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.zeros(len(X), dtype=int) # Assign to cluster 0
            
    # We need to mock it where it is imported/used. 
    # Since it's imported inside the method, we might need to mock sys.modules or use a different approach.
    # But wait, the import is inside `_assign_noise_knn`.
    # Let's just mock `_assign_noise_knn` to keep it simple if we don't want to test sklearn.
    
    def mock_assign_noise(self, vectors, labels):
        # Simple assignment: turn all -1 to 0
        labels[labels == -1] = 0
        return labels
        
    monkeypatch.setattr(main.VideoClusterer, "_assign_noise_knn", mock_assign_noise)

    return TestClient(main.app)


def _sample_payload(count=6):
    payload = []
    for idx in range(count):
        payload.append(
            {
                "id": f"video-{idx}",
                "vector": [0.1 * idx] * 10  # Simple dummy vector
            }
        )
    return payload


def test_cluster_endpoint_success(cluster_client):
    response = cluster_client.post(
        "/api/cluster",
        json=_sample_payload(),
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "cluster_id" in data[0]
    assert "video_ids" in data[0]
    assert len(data) > 0


def test_cluster_endpoint_requires_auth(cluster_client):
    response = cluster_client.post("/api/cluster", json=_sample_payload())
    assert response.status_code == 401


def test_cluster_endpoint_validates_minimum_payload(cluster_client):
    response = cluster_client.post(
        "/api/cluster",
        json=_sample_payload(4),
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400


def test_cluster_endpoint_validates_vector_type(cluster_client):
    payload = _sample_payload(6)
    payload[0]["vector"] = ["not", "a", "number"]
    response = cluster_client.post(
        "/api/cluster",
        json=payload,
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400


def test_cluster_endpoint_secondary_clustering(monkeypatch):
    """
    Test that secondary clustering is triggered when noise is high.
    """
    main.AUTH_KEY = "test-key"
    
    call_count = 0

    class MockHighNoiseClusterer:
        def __init__(self, *args, **kwargs):
            pass

        def fit_predict(self, vectors):
            nonlocal call_count
            call_count += 1
            n = len(vectors)
            if call_count == 1:
                # First pass: all noise
                return np.full(n, -1, dtype=int)
            else:
                # Second pass (secondary clustering): return valid clusters
                labels = np.zeros(n, dtype=int)
                labels[n // 2 :] = 1
                return labels

    monkeypatch.setattr(main.hdbscan, "HDBSCAN", MockHighNoiseClusterer)

    # Mock helper methods on the class
    def identity_reduce(self, vectors):
        return vectors
    monkeypatch.setattr(main.VideoClusterer, "_reduce_and_normalize", identity_reduce)
    
    def mock_assign_noise(self, vectors, labels):
        labels[labels == -1] = 0
        return labels
    monkeypatch.setattr(main.VideoClusterer, "_assign_noise_knn", mock_assign_noise)
    
    client = TestClient(main.app)
    
    # We need enough videos to trigger the logic
    payload = _sample_payload(10)
    
    response = client.post(
        "/api/cluster",
        json=payload,
        headers={"Authorization": "Bearer test-key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # HDBSCAN should be called twice (initial + secondary)
    assert call_count == 2
    assert len(data) > 0


def test_iterative_fallback(monkeypatch):
    """
    Test that we iteratively split clusters if we are below MIN_CLUSTERS.
    """
    main.AUTH_KEY = "test-key"
    
    # Mock HDBSCAN to return just 2 clusters initially
    class MockFewClusters:
        def __init__(self, *args, **kwargs):
            pass
        def fit_predict(self, vectors):
            # Return 2 clusters: 0 and 1
            n = len(vectors)
            labels = np.zeros(n, dtype=int)
            labels[n // 2 :] = 1
            return labels

    monkeypatch.setattr(main.hdbscan, "HDBSCAN", MockFewClusters)

    # Mock KMeans to split clusters
    # We need to mock sklearn.cluster.KMeans used in _perform_kmeans (and _ensure_min_clusters)
    class MockKMeans:
        def __init__(self, n_clusters=2, **kwargs):
            self.n_clusters = n_clusters
            
        def fit_predict(self, vectors):
            # Split whatever we get into n_clusters (usually 2 for split)
            n = len(vectors)
            labels = np.zeros(n, dtype=int)
            if n > 1:
                labels[n // 2 :] = 1
            return labels

    monkeypatch.setattr(main, "KMeans", MockKMeans)

    # Mock other helpers
    def identity_reduce(self, vectors):
        return vectors
    monkeypatch.setattr(main.VideoClusterer, "_reduce_and_normalize", identity_reduce)
    
    def mock_assign_noise(self, vectors, labels):
        return labels
    monkeypatch.setattr(main.VideoClusterer, "_assign_noise_knn", mock_assign_noise)

    client = TestClient(main.app)
    
    # 10 videos. MIN_CLUSTERS is 5.
    # Initial: 2 clusters.
    # Iteration 1: Split largest (say 0) -> +1 cluster = 3.
    # Iteration 2: Split largest -> +1 cluster = 4.
    # Iteration 3: Split largest -> +1 cluster = 5.
    # Should stop.
    
    payload = _sample_payload(20) # More videos to allow splitting
    
    response = client.post(
        "/api/cluster",
        json=payload,
        headers={"Authorization": "Bearer test-key"},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check that we have at least 5 clusters
    unique_clusters = set(item["cluster_id"] for item in data)
    assert len(unique_clusters) >= 5
