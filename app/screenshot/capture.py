import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import re
from dataclasses import dataclass
from PIL import Image
import io

HARD_TIMEOUT_SECONDS = 30
NETWORK_IDLE_TIMEOUT_MS = 15000

@dataclass
class ScreenshotResult:
    """Dataclass to hold screenshot results."""
    image_bytes: bytes
    width: int
    height: int

class ScreenshotError(Exception):
    """Custom exception for screenshot failures."""
    pass

async def capture_full_page(url: str) -> ScreenshotResult:
    """
    Captures a full-page screenshot of a given URL, handling timeouts and pop-ups.
    
    Args:
        url: The URL to capture.
        
    Returns:
        The screenshot image in bytes.

    Raises:
        ScreenshotError: If any step of the capture process fails or times out.
    """
    async def capture_logic():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
            
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=NETWORK_IDLE_TIMEOUT_MS)

                # Use get_by_text with a regex to find and click the accept button
                try:
                    accept_button = page.get_by_text(re.compile(r"Accept|Agree|Got it|OK|I agree", re.IGNORECASE)).first
                    await accept_button.click(timeout=1500)
                    await page.wait_for_timeout(500)
                except PlaywrightTimeoutError:
                    pass # Button not found or did not need clicking

                hide_css = """
                    [id*="cookie"], [class*="cookie"], [id*="consent"], [class*="consent"],
                    [id*="banner"], [class*="banner"], [role="dialog"], [aria-modal="true"] {
                        display: none !important; visibility: hidden !important;
                    }
                """
                await page.add_style_tag(content=hide_css)
                await page.wait_for_timeout(200)

                # Scroll to the bottom of the page
                last_height = await page.evaluate("document.body.scrollHeight")
                while True:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1000)
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                
                await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)

                screenshot_bytes = await page.screenshot(full_page=True)
                
                # Get image dimensions using Pillow
                with Image.open(io.BytesIO(screenshot_bytes)) as img:
                    width, height = img.size

                return ScreenshotResult(image_bytes=screenshot_bytes, width=width, height=height)
            finally:
                if 'browser' in locals() and browser.is_connected():
                    await browser.close()

    try:
        return await asyncio.wait_for(capture_logic(), timeout=HARD_TIMEOUT_SECONDS)
    except (PlaywrightTimeoutError, asyncio.TimeoutError) as e:
        raise ScreenshotError(f"Screenshot failed for {url} due to timeout.") from e
    except Exception as e:
        raise ScreenshotError(f"An unexpected error occurred while capturing {url}: {e}") from e 