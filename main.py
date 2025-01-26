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
    loader = YoutubeLoader.from_youtube_url(
        request.url,
        # add_video_info=True,
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=60,
        language=["en", "en-US", "es", "es-ES", "zh", "zh-CN", "de", "de-DE", "fr", "fr-FR", "ar", "ar-SA"],
    )
    docs = loader.load()
    return docs

class S3DocumentRequest(BaseModel):
    key: str
    id: str
    name: str

@app.post("/parse-document/")
async def get_chunked_ocr(request: S3DocumentRequest):
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Get file from S3
        s3_response = s3_client.get_object(
            Bucket=os.getenv('BUCKET_NAME'),
            Key=request.key
        )
        
        # Create temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, request.name)
            
            # Write S3 content to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(s3_response['Body'].read())
            
            # Initialize LlamaParse
            parser = LlamaParse(
                api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
                result_type="markdown",
              premium_mode=True,
                webhook_url=f"https://www.studyguidemaker.com/api/llama?id={request.id}&name={request.name}",
               split_by_page=True
            )
            
            # Process the file
            file_extractor = {".pdf": parser}
            documents = SimpleDirectoryReader(
                input_files=[temp_file_path], 
                file_extractor=file_extractor
            ).load_data()
            
            return documents
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException (status_code=500, detail="File docs chunking failed!")

class MarkdownRequest(BaseModel):
    key: str
    name: str

@app.post("/convert-markdown/")
async def convert_to_markdown(request: MarkdownRequest):
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Get file from S3
        s3_response = s3_client.get_object(
            Bucket="jenga",
            Key=request.key
        )
        
        # Create temporary file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, request.name)
            
            # Write S3 content to temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(s3_response['Body'].read())
            
            # Initialize MarkItDown and convert
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
