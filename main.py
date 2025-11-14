from collections import Counter
from typing import List

from fastapi import FastAPI, HTTPException, Request
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import os
from dotenv import load_dotenv
import boto3
import tempfile
import json
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from markitdown import MarkItDown
from datetime import datetime, timezone
import logging
import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Prefer GOOGLE_GENERATIVE_AI_API_KEY, fall back to GOOGLE_API_KEY if present
GOOGLE_GENAI_KEY = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get(
    "GOOGLE_API_KEY"
)


class Topic(BaseModel):
    """A structured representation of a video topic cluster."""

    topic_name: str = Field(description="A short, 3-word-or-less title for this topic")
    topic_description: str = Field(description="A description of what this topic is about.")


print("Initializing models...")
try:
    embedding_client = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GENAI_KEY,
    )
    print("Embeddings client ready.")
except Exception as e:  # pragma: no cover - guard rail for deployment misconfig
    print(f"FATAL: Could not configure Google embeddings: {e}")
    embedding_client = None


try:
    # IMPORTANT: Set GOOGLE_GENERATIVE_AI_API_KEY (or GOOGLE_API_KEY) in Vercel.
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash",
        temperature=0,
        google_api_key=GOOGLE_GENAI_KEY,
    )
    structured_llm = llm.with_structured_output(
        schema=Topic.model_json_schema(), method="json_schema"
    )
    print("LLM initialized successfully.")
except Exception as e:  # pragma: no cover - guard rail for deployment misconfig
    print(f"FATAL: Could not configure LangChain/Gemini model: {e}")
    structured_llm = None


# IMPORTANT: Set your VERCEL_API_KEY as an environment variable in Vercel for security.
AUTH_KEY = os.environ.get("VERCEL_API_KEY")


def _coerce_string_list(values: List, field_name: str, index: int) -> List[str]:
    """Ensure lists contain clean, comparable string tokens."""
    if not isinstance(values, list):
        raise ValueError(f"Field '{field_name}' in video object at index {index} must be a list.")

    coerced: List[str] = []
    for item in values:
        if item is None:
            continue
        if not isinstance(item, str):
            try:
                item = str(item)
            except Exception as exc:
                raise ValueError(
                    f"Field '{field_name}' in video object at index {index} contains non-string items."
                ) from exc
        stripped = item.strip()
        if stripped:
            coerced.append(stripped)
    return coerced


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Batch texts through the Google embeddings API and return numpy arrays."""
    if embedding_client is None:
        raise RuntimeError("Embedding client is not configured.")

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    embeddings: List[List[float]] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        normalized_chunk = [text if text.strip() else " " for text in chunk]
        embeddings.extend(embedding_client.embed_documents(normalized_chunk))

    return np.asarray(embeddings, dtype=np.float32)


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


def _summarize_cluster(videos: List[dict]) -> dict:
    keyword_counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()
    visual_counter: Counter[str] = Counter()

    for video in videos:
        keyword_counter.update(video.get("keywords", []))
        phrase_counter.update(video.get("key_phrases", []))
        visuals = [
            token.strip()
            for token in video.get("visual_elements", "").split(",")
            if token.strip()
        ]
        visual_counter.update(visuals or [video.get("visual_elements", "")])

    def top_values(counter: Counter[str]) -> List[str]:
        return [term for term, _ in counter.most_common(8)]

    return {
        "keywords": top_values(keyword_counter),
        "key_phrases": top_values(phrase_counter),
        "visual_elements": top_values(visual_counter),
    }

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
    Main endpoint to receive video data, perform clustering, and return named topics.
    This function is the request handler; it runs for every API call.
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

    if not structured_llm:
        return JSONResponse(
            {
                "error": "LLM model is not configured. Check server logs and GOOGLE_GENERATIVE_AI_API_KEY/GOOGLE_API_KEY.",
            },
            status_code=500,
        )
    if embedding_client is None:
        return JSONResponse(
            {
                "error": "Embedding model is not configured. Check server logs and GOOGLE_GENERATIVE_AI_API_KEY/GOOGLE_API_KEY."
            },
            status_code=500,
        )

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
            "summary": str,
            "keywords": list,
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

            video["keywords"] = _coerce_string_list(video["keywords"], "keywords", i)
            raw_key_phrases = video.get("key_phrases")
            if raw_key_phrases is None:
                video["key_phrases"] = []
            else:
                if not isinstance(raw_key_phrases, list):
                    return JSONResponse(
                        {
                            "error": f"Field 'key_phrases' at index {i} has incorrect type. Expected list."
                        },
                        status_code=400,
                    )
                video["key_phrases"] = _coerce_string_list(raw_key_phrases, "key_phrases", i)

            raw_visuals = video.get("visual_elements")
            if raw_visuals is None:
                video["visual_elements"] = ""
            else:
                if not isinstance(raw_visuals, str):
                    return JSONResponse(
                        {
                            "error": f"Field 'visual_elements' at index {i} has incorrect type. Expected str."
                        },
                        status_code=400,
                    )
                video["visual_elements"] = raw_visuals
    except ValueError as value_error:
        logger.warning("Cluster request validation error.", extra={"error": str(value_error)})
        return JSONResponse({"error": str(value_error)}, status_code=400)
    except Exception as exc:
        logger.warning("Cluster request JSON parsing failed.", extra={"error": str(exc)})
        return JSONResponse({"error": "Invalid or malformed JSON in request body."}, status_code=400)

    print(f"Processing {len(videos_data)} videos...")
    base_docs: List[str] = []
    enriched_docs: List[str] = []
    video_metadata: List[dict] = []
    for video in videos_data:
        keywords_text = " ".join(video["keywords"])
        base_doc = f"{video['summary']} {keywords_text} {video['visual_elements']}".strip()
        key_phrase_doc = " ".join(video["key_phrases"]) or video["summary"]
        base_docs.append(base_doc)
        enriched_docs.append(key_phrase_doc)
        video_metadata.append(
            {
                "id": video["id"],
                "summary": video["summary"],
                "keywords": video["keywords"],
                "key_phrases": video["key_phrases"],
                "visual_elements": video["visual_elements"],
            }
        )

    BASE_WEIGHT = 0.7
    PHRASE_WEIGHT = 0.3
    base_vectors = _embed_texts(base_docs)
    phrase_vectors = _embed_texts(enriched_docs)
    all_vectors_np = np.hstack([base_vectors * BASE_WEIGHT, phrase_vectors * PHRASE_WEIGHT])
    all_vectors_np = _reduce_and_normalize(all_vectors_np)

    min_cluster_size = max(5, int(len(videos_data) * 0.03))
    min_samples = max(2, int(min_cluster_size * 0.6))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
    )
    cluster_labels = clusterer.fit_predict(all_vectors_np)
    print(
        f"Clustering complete. Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} topics."
    )

    clustered_videos = {}
    for i, label in enumerate(cluster_labels):
        if label != -1:
            clustered_videos.setdefault(label, []).append(video_metadata[i])

    if not clustered_videos:
        return JSONResponse(
            {"error": "No stable clusters could be formed from the provided videos."}, status_code=422
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert market analyst. Name the dominant topic reflected by the cluster.",
            ),
            (
                "human",
                (
                    "Given the following information, output a concise title (≤3 words) and a one-sentence "
                    "description capturing the common theme.\n"
                    "Summaries:\n---\n{summaries}\n---\n"
                    "Top keywords: {keywords}\n"
                    "Top key phrases: {key_phrases}\n"
                    "Visual elements: {visual_elements}"
                ),
            ),
        ]
    )
    chain = prompt | structured_llm

    final_topics = []
    for label, videos in clustered_videos.items():
        summaries_text = "\n---\n".join([v["summary"] for v in videos[:50]])
        cluster_stats = _summarize_cluster(videos)
        try:
            topic_payload = chain.invoke(
                {
                    "summaries": summaries_text,
                    "keywords": ", ".join(cluster_stats["keywords"]) or "n/a",
                    "key_phrases": ", ".join(cluster_stats["key_phrases"]) or "n/a",
                    "visual_elements": ", ".join(cluster_stats["visual_elements"]) or "n/a",
                }
            )
            topic_object = Topic(**topic_payload)
            final_topics.append(
                {
                    "topic_name": topic_object.topic_name,
                    "topic_description": topic_object.topic_description,
                    "video_ids": [v["id"] for v in videos],
                }
            )
        except ValidationError as validation_error:
            print(f"Validation error on cluster {label}: {validation_error}")
            final_topics.append(
                {
                    "topic_name": f"Invalid Topic {label}",
                    "topic_description": "LLM output was not in the expected schema.",
                    "video_ids": [v["id"] for v in videos],
                }
            )
        except OutputParserException as parser_error:
            print(f"Parser error on cluster {label}: {parser_error}")
            final_topics.append(
                {
                    "topic_name": f"Unparsed Topic {label}",
                    "topic_description": "LLM output could not be parsed reliably.",
                    "video_ids": [v["id"] for v in videos],
                }
            )
        except Exception as e:
            print(f"Error processing cluster {label}: {e}")
            final_topics.append(
                {
                    "topic_name": f"Error Topic {label}",
                    "topic_description": "Could not generate a name for this topic due to a processing error.",
                    "video_ids": [v["id"] for v in videos],
                }
            )

    print("Processing complete.")
    return JSONResponse(final_topics)


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
