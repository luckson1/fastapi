from collections import Counter
from typing import List

from fastapi import FastAPI, HTTPException, Request
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import os
import boto3
import tempfile
import json
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from markitdown import MarkItDown
from datetime import datetime, timezone
from dotenv import load_dotenv
import logging
import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load environment variables early for local development.
load_dotenv()

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Minimum number of clusters to return
MIN_CLUSTERS = 5


# IMPORTANT: Set your VERCEL_API_KEY as an environment variable in Vercel for security.
AUTH_KEY = os.environ.get("AUTH_KEY")


def _reduce_and_normalize(vectors: np.ndarray) -> np.ndarray:
    """Drop the dimensionality and normalize, improving HDBSCAN stability."""
    n_samples, n_features = vectors.shape
    if n_samples >= 15 and n_features > 100:
        max_components = min(100, n_features, n_samples - 1)
        if max_components >= 10:
            pca = PCA(n_components=max_components, random_state=42)
            vectors = pca.fit_transform(vectors)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class YouTubeTranscriptRequest(BaseModel):
    url: str

app = FastAPI()

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

@app.post("/api/cluster")
async def cluster_videos(request: Request):
    """
    Main endpoint to receive video embeddings, perform clustering, and return cluster assignments.
    """
    auth_header = request.headers.get("Authorization")
    log_context = {"auth_header": auth_header or "<missing>"}
    if not AUTH_KEY:
        logger.error("AUTH_KEY is not configured; rejecting cluster request.", extra=log_context)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    expected_header = f"Bearer {AUTH_KEY}"
    if auth_header != expected_header:
        logger.warning("Cluster request provided invalid auth.", extra=log_context)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    logger.info("Cluster request authorized.", extra=log_context)

    try:
        videos_data = await request.json()
        payload_preview = _serialize_payload_for_logging(videos_data)
        logger.info(
            "Cluster request payload parsed.",
            extra={
                "video_count": len(videos_data) if isinstance(videos_data, list) else "non-list",
                "payload_preview": payload_preview,
            },
        )
        if not isinstance(videos_data, list) or len(videos_data) < 5:
            return JSONResponse(
                {"error": "Request body must be a JSON array of at least 5 video objects."},
                status_code=400,
            )

        required_fields = {
            "id": str,
            "vector": list,
        }
        for i, video in enumerate(videos_data):
            for field, field_type in required_fields.items():
                if field not in video:
                    return JSONResponse(
                        {"error": f"Missing field '{field}' in video object at index {i}."},
                        status_code=400,
                    )
                if not isinstance(video[field], field_type):
                    return JSONResponse(
                        {
                            "error": f"Field '{field}' at index {i} has incorrect type. Expected {field_type.__name__}."
                        },
                        status_code=400,
                    )
            
            # Validate vector contents
            if not all(isinstance(x, (int, float)) for x in video["vector"]):
                 return JSONResponse(
                        {"error": f"Field 'vector' at index {i} must contain only numbers."},
                        status_code=400,
                    )

    except ValueError as value_error:
        logger.warning("Cluster request validation error.", extra={"error": str(value_error)})
        return JSONResponse({"error": str(value_error)}, status_code=400)
    except Exception as exc:
        logger.warning("Cluster request JSON parsing failed.", extra={"error": str(exc)})
        return JSONResponse({"error": "Invalid or malformed JSON in request body."}, status_code=400)

    print(f"Processing {len(videos_data)} videos...")
    
    # Extract vectors and IDs
    vectors_list = [v["vector"] for v in videos_data]
    video_ids = [v["id"] for v in videos_data]
    
    all_vectors_np = np.array(vectors_list, dtype=np.float32)
    all_vectors_np = _reduce_and_normalize(all_vectors_np)

    min_cluster_size = max(5, int(len(videos_data) * 0.03))
    min_samples = max(2, int(min_cluster_size * 0.6))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    cluster_labels = clusterer.fit_predict(all_vectors_np)

    # Ensure at least MIN_CLUSTERS clusters by falling back to KMeans if needed
    non_noise_cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    if non_noise_cluster_count < MIN_CLUSTERS:
        desired_k = min(len(videos_data), MIN_CLUSTERS)
        kmeans = KMeans(n_clusters=desired_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(all_vectors_np)
        print(f"HDBSCAN produced {non_noise_cluster_count} clusters; fell back to KMeans with k={desired_k}.")

    final_cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Clustering complete. Found {final_cluster_count} topics.")

    clustered_videos = {}
    for i, label in enumerate(cluster_labels):
        if label != -1:
            clustered_videos.setdefault(int(label), []).append(video_ids[i])

    if not clustered_videos:
        return JSONResponse(
            {"error": "No stable clusters could be formed from the provided videos."}, status_code=422
        )

    final_response = []
    for label, v_ids in clustered_videos.items():
        final_response.append(
            {
                "cluster_id": label,
                "video_ids": v_ids,
            }
        )

    print("Processing complete.")
    return JSONResponse(final_response)


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
def _serialize_payload_for_logging(payload, limit: int = 2000) -> str:
    """Serialize payloads for logging without overwhelming the log stream."""
    try:
        text = json.dumps(payload)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > limit:
        return f"{text[:limit]}...(truncated)"
    return text

