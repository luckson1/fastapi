from fastapi import FastAPI, HTTPException, status, Depends
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, HttpUrl, Field
import os
from dotenv import load_dotenv
import boto3
import tempfile
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from markitdown import MarkItDown
from datetime import datetime, timezone
from app.screenshot.capture import capture_full_page, ScreenshotError
from app.storage.s3 import S3Storage
from app.metrics.prom import (
    SCREENSHOT_REQUESTS_TOTAL,
    SCREENSHOT_SUCCESS_TOTAL,
    SCREENSHOT_FAILURE_TOTAL,
    SCREENSHOT_LATENCY_SECONDS,
)
import time
from prometheus_client import make_asgi_app
import logging

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a separate app for the metrics endpoint
metrics_app = make_asgi_app()

# Dependency to get S3 storage instance
def get_s3_storage():
    return S3Storage()

class YouTubeTranscriptRequest(BaseModel):
    url: str

class ScreenshotRequest(BaseModel):
    url: HttpUrl = Field(..., max_length=2048)

app = FastAPI()
app.mount("/metrics", metrics_app)

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/youtube/")
def load_youtube_transcript(request: YouTubeTranscriptRequest):
    try:
        loader = YoutubeLoader.from_youtube_url(
            request.url,
            # add_video_info=True,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=60,
            language=["en", "en-US", "es", "es-ES", "zh", "zh-CN", "de", "de-DE", "fr", "fr-FR", "ar", "ar-SA"],
        )
        docs = loader.load()
     
        return docs
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load YouTube transcript")

class MarkdownRequest(BaseModel):
    key: str
    name: str

@app.post("/convert-markdown/")
async def convert_to_markdown(request: MarkdownRequest):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        s3_response = s3_client.get_object(
            Bucket=os.getenv('BUCKET_NAME'),
            Key=request.key
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, request.name)

            # Stream file directly to disk
            with open(temp_file_path, 'wb') as f:
                for chunk in s3_response['Body'].iter_chunks(chunk_size=8192):
                    f.write(chunk)

            md = MarkItDown()
            result = md.convert(temp_file_path)

            return {"markdown_content": result.text_content}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Markdown conversion failed!")

@app.post("/v1/screenshot", status_code=status.HTTP_201_CREATED)
async def take_screenshot(request: ScreenshotRequest, storage: S3Storage = Depends(get_s3_storage)):
    """
    Accepts a URL, validates it, captures a screenshot, and stores it in S3.
    """
    SCREENSHOT_REQUESTS_TOTAL.inc()
    start_time = time.time()
    log_extra = {"url": str(request.url)}

    try:
        # 1. Capture the screenshot
        logger.info("Starting screenshot capture", extra=log_extra)
        result = await capture_full_page(str(request.url))

        # 2. Store in S3
        s3_key = storage.upload_file(result.image_bytes)
        log_extra["s3_key"] = s3_key
        logger.info("Screenshot capture and upload successful", extra=log_extra)

        # 3. Return the structured response
        SCREENSHOT_SUCCESS_TOTAL.inc()
        return {
            "s3_key": s3_key,
            "width": result.width,
            "height": result.height,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }

    except ScreenshotError as e:
        log_extra["error"] = str(e)
        logger.error("Screenshot capture failed", extra=log_extra)
        SCREENSHOT_FAILURE_TOTAL.inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Screenshot capture failed: {e}")
    except Exception as e:
        log_extra["error"] = str(e)
        logger.error("An unexpected error occurred", extra=log_extra)
        SCREENSHOT_FAILURE_TOTAL.inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")
    finally:
        latency = time.time() - start_time
        log_extra["latency_seconds"] = latency
        logger.info("Screenshot request finished", extra=log_extra)
        SCREENSHOT_LATENCY_SECONDS.observe(latency)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "link": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
