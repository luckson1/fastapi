import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class DummyPromptTemplate:
    @classmethod
    def from_messages(cls, _messages):
        class DummyPrompt:
            def __or__(self, _structured_llm):
                class DummyChain:
                    def invoke(self_inner, inputs):
                        return {
                            "topic_name": "Dummy Topic",
                            "topic_description": f"Keywords: {inputs['keywords']}",
                        }

                return DummyChain()

        return DummyPrompt()


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
    main.structured_llm = object()
    main.embedding_client = object()
    monkeypatch.setattr(main, "ChatPromptTemplate", DummyPromptTemplate)
    monkeypatch.setattr(main.hdbscan, "HDBSCAN", DummyClusterer)

    def fake_embed_texts(texts):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4)

    def identity_reduce(vectors):
        return vectors

    monkeypatch.setattr(main, "_embed_texts", fake_embed_texts)
    monkeypatch.setattr(main, "_reduce_and_normalize", identity_reduce)
    return TestClient(main.app)


def _sample_payload(count=6):
    payload = []
    for idx in range(count):
        payload.append(
            {
                "id": f"video-{idx}",
                "summary": f"Summary {idx}",
                "keywords": [f"keyword-{idx}", "shared"],
                "key_phrases": [f"phrase-{idx}"],
                "visual_elements": "visual",
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
    assert data[0]["topic_name"] == "Dummy Topic"
    assert all("video_ids" in topic for topic in data)


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


def test_cluster_endpoint_llm_not_configured(cluster_client, monkeypatch):
    monkeypatch.setattr(main, "structured_llm", None)
    response = cluster_client.post(
        "/api/cluster",
        json=_sample_payload(),
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 500
