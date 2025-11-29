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


class VideoClusterer:
    def __init__(self, min_clusters: int = 5):
        self.min_clusters = min_clusters

    def _reduce_and_normalize(self, vectors: np.ndarray) -> np.ndarray:
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

    def _perform_hdbscan(self, vectors: np.ndarray, min_cluster_size: int = None) -> np.ndarray:
        """Run HDBSCAN on the given vectors."""
        if min_cluster_size is None:
            # Cap the growth: max(5, min(20, 3% of data))
            min_cluster_size = max(5, min(20, int(len(vectors) * 0.03)))
        
        min_samples = max(2, int(min_cluster_size * 0.6))
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        return clusterer.fit_predict(vectors)

    def _perform_kmeans(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """Run KMeans on the given vectors."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        return kmeans.fit_predict(vectors)

    def _assign_noise_knn(self, vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Assign noise points (-1) to the nearest labeled neighbor using KNN."""
        from sklearn.neighbors import KNeighborsClassifier

        mask_noise = labels == -1
        if not np.any(mask_noise):
            return labels
            
        mask_valid = ~mask_noise
        # If we have no valid clusters, we can't assign noise to anything.
        if not np.any(mask_valid):
            return labels

        X_train = vectors[mask_valid]
        y_train = labels[mask_valid]
        X_test = vectors[mask_noise]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        predicted_labels = knn.predict(X_test)
        
        labels[mask_noise] = predicted_labels
        return labels

    def _ensure_min_clusters(self, vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Iteratively split the largest clusters until we reach MIN_CLUSTERS.
        This preserves existing structure while forcing granularity where needed.
        """
        current_cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Safety break to prevent infinite loops
        max_iterations = self.min_clusters + 5
        iteration = 0

        while current_cluster_count < self.min_clusters and iteration < max_iterations:
            iteration += 1
            
            # Find the largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) == 0:
                break
                
            largest_cluster_label = unique_labels[np.argmax(counts)]
            mask_largest = labels == largest_cluster_label
            points_in_cluster = vectors[mask_largest]
            
            if len(points_in_cluster) < 2:
                # Can't split a singleton or empty cluster
                break
                
            # Split into 2 sub-clusters
            sub_labels = self._perform_kmeans(points_in_cluster, k=2)
            
            # Reassign labels
            # One part keeps the old label, the other gets a new max label
            new_label = np.max(labels) + 1
            
            # sub_labels will be 0s and 1s. 
            # 0s keep 'largest_cluster_label', 1s get 'new_label'
            # We need to map this back to the original array indices
            indices_to_update = np.where(mask_largest)[0]
            
            # Update only those that are '1' in the sub-clustering
            indices_for_new_label = indices_to_update[sub_labels == 1]
            labels[indices_for_new_label] = new_label
            
            current_cluster_count += 1
            print(f"Split cluster {largest_cluster_label} (size {len(points_in_cluster)}) into 2. New count: {current_cluster_count}")

        return labels

    def cluster(self, videos_data: List[dict]) -> List[dict]:
        print(f"Processing {len(videos_data)} videos...")
        
        # Extract vectors and IDs
        vectors_list = [v["vector"] for v in videos_data]
        video_ids = [v["id"] for v in videos_data]
        
        all_vectors_np = np.array(vectors_list, dtype=np.float32)
        all_vectors_np = self._reduce_and_normalize(all_vectors_np)

        # 1. Initial HDBSCAN
        cluster_labels = self._perform_hdbscan(all_vectors_np)

        # 2. Calculate Noise Ratio
        noise_indices = [i for i, l in enumerate(cluster_labels) if l == -1]
        noise_ratio = len(noise_indices) / len(all_vectors_np)
        
        # 3. Conditional Secondary Clustering
        # Lowered threshold to 0.25 for earlier intervention
        if noise_ratio > 0.25:
            print(f"High noise detected ({noise_ratio:.1%}). Re-clustering noise...")
            
            noise_vectors = all_vectors_np[noise_indices]
            
            # Secondary pass: aim for small clusters
            secondary_min_cluster_size = max(3, len(noise_vectors) // 4)
            secondary_labels = self._perform_hdbscan(noise_vectors, min_cluster_size=secondary_min_cluster_size)
            
            # Merge labels back
            max_label = np.max(cluster_labels) if np.any(cluster_labels != -1) else -1
            for i, idx in enumerate(noise_indices):
                sec_label = secondary_labels[i]
                if sec_label != -1:
                    cluster_labels[idx] = sec_label + max_label + 1

        # 4. Assign remaining noise using KNN
        cluster_labels = self._assign_noise_knn(all_vectors_np, cluster_labels)

        # 5. Ensure Minimum Clusters (Iterative Split)
        cluster_labels = self._ensure_min_clusters(all_vectors_np, cluster_labels)

        final_cluster_count = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Clustering complete. Found {final_cluster_count} topics.")

        clustered_videos = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:
                clustered_videos.setdefault(int(label), []).append(video_ids[i])

        final_response = []
        for label, v_ids in clustered_videos.items():
            final_response.append(
                {
                    "cluster_id": label,
                    "video_ids": v_ids,
                }
            )
            
        return final_response


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

    try:
        clusterer = VideoClusterer(min_clusters=MIN_CLUSTERS)
        result = clusterer.cluster(videos_data)
        
        if not result:
             return JSONResponse(
                {"error": "No stable clusters could be formed from the provided videos."}, status_code=422
            )
            
        print("Processing complete.")
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Clustering failed: {str(e)}")
        return JSONResponse({"error": f"Clustering failed: {str(e)}"}, status_code=500)


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

