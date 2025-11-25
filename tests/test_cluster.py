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
    monkeypatch.setattr(main.hdbscan, "HDBSCAN", DummyClusterer)

    def identity_reduce(vectors):
        return vectors

    monkeypatch.setattr(main, "_reduce_and_normalize", identity_reduce)
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
