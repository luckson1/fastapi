from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import boto3
import tempfile
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from markitdown import MarkItDown

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
