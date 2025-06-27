import pytest
from pathlib import Path
from app.screenshot.capture import capture_full_page, ScreenshotError, ScreenshotResult

# Make sure to run `pytest` from the root of the project
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "test_page.html"

@pytest.mark.asyncio
async def test_capture_successful():
    """
    Tests a successful screenshot capture of a local HTML file.
    """
    url = f"file://{FIXTURE_PATH.resolve()}"
    result = await capture_full_page(url)
    
    assert result is not None
    assert isinstance(result, ScreenshotResult)
    assert result.width > 0
    assert result.height > 0
    assert isinstance(result.image_bytes, bytes)
    # A simple check for PNG header
    assert result.image_bytes.startswith(b'\x89PNG\r\n\x1a\n')

@pytest.mark.asyncio
async def test_capture_timeout_error():
    """
    Tests that a ScreenshotError is raised for a URL that will time out.
    Playwright's default navigation timeout is 30s, so we use a non-routable address.
    """
    # This IP address is reserved for documentation and should not be routable.
    url = "http://192.0.2.1"
    
    with pytest.raises(ScreenshotError, match="timeout"):
        await capture_full_page(url)

@pytest.mark.asyncio
async def test_capture_invalid_url():
    """
    Tests that a ScreenshotError is raised for an invalid URL.
    """
    url = "not-a-valid-url"
    
    # Playwright's goto raises its own error for malformed URLs, which our wrapper catches.
    with pytest.raises(ScreenshotError, match="An unexpected error occurred"):
        await capture_full_page(url) 