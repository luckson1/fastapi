import pytest
from fastapi.testclient import TestClient
from main import app, get_s3_storage
from app.screenshot.capture import ScreenshotResult
from app.storage.s3 import S3Storage

# Mock S3Storage for testing
class MockS3Storage:
    def upload_file(self, file_bytes: bytes, content_type: str = "image/png") -> str:
        return "screenshots/mock-key.png"

def get_mock_s3_storage():
    return MockS3Storage()

app.dependency_overrides[get_s3_storage] = get_mock_s3_storage

client = TestClient(app)

@pytest.fixture
def mock_capture(monkeypatch):
    """Mocks the screenshot capture function."""
    async def mock_capture_func(url: str):
        return ScreenshotResult(
            image_bytes=b"fake-image-bytes",
            width=1920,
            height=1080
        )
    monkeypatch.setattr("main.capture_full_page", mock_capture_func)

def test_screenshot_endpoint_success(mock_capture):
    """
    Tests the /v1/screenshot endpoint for a successful request.
    """
    response = client.post("/v1/screenshot", json={"url": "https://example.com"})
    
    assert response.status_code == 201
    data = response.json()
    assert data["s3_key"] == "screenshots/mock-key.png"
    assert data["width"] == 1920
    assert data["height"] == 1080
    assert "captured_at" in data

def test_screenshot_endpoint_invalid_url():
    """
    Tests the /v1/screenshot endpoint with an invalid URL.
    """
    response = client.post("/v1/screenshot", json={"url": "not-a-valid-url"})
    
    assert response.status_code == 422 