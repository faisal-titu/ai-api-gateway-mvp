# Standard library imports
import argparse
import glob
import io
import logging
import os
import tempfile
from typing import List, Optional


# Third-party library imports
import PIL
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader

# Local application imports
from dev.image_embedding.create_image_index import create_index
from dev.image_embedding.embedding_generation import get_image_embedding, get_text_embedding
from dev.image_embedding.image_batch_embedding import *
from dev.image_embedding.open_search_client import client
from dev.image_embedding.search import search_knn
from dev.text_embedding.create_text_index import create_index_text
from dev.text_embedding.text_batch_embedding import (
    bulk_index_text_embeddings,
    create_text_dataloader,
    generate_text_embeddings,
)
from dev.face_embedding.face_detection import detect_and_crop_faces
from dev.face_embedding.face_dataloader import CroppedFaceDataset
from dev.face_embedding.face_embedding_generation import generate_embeddings_and_index
from dev.face_embedding.single_image_processing import detect_faces_in_image, generate_face_embeddings

# Initialize FastAPI app
app = FastAPI(title="AI Search API")

# Index and image directory configuration
INDEX_NAME = ''
IMAGE_DIR = ""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("/home/iot/Desktop/ai_search_fastapi/ai-api-gateway-mvp/logs/app.log")  # Log to a file
    ]
)
logger = logging.getLogger(__name__)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods
    allow_headers=["*"],    # Allows all headers
)

# Routers
image_router = APIRouter(prefix="/images", tags=["Image Operations"])
text_router = APIRouter(prefix="/texts", tags=["Text Operations"])
face_router = APIRouter(prefix="/faces", tags=["Face Operations"])

# Define the input data model
class TextQueryRequest(BaseModel):
    query: str
    num_images: int = 5

# Define the response model
class SearchResult(BaseModel):
    image_ids: List[str]

# Define the response model for image search results
class ImageSearchResult(BaseModel):
    query_image_id: str
    similar_image_ids: List[str]

class ImageSearchRequest(BaseModel):
    num_images: int = 5

# Endpoint to dynamically set index name and image directory
class SettingsRequest(BaseModel):
    index_name: str
    image_dir: str

class FaceDetectionResult(BaseModel):
    processed_images: List[str]

@app.post("/set-settings")
async def set_settings(settings: SettingsRequest):
    global INDEX_NAME, IMAGE_DIR
    logger.info(f"Setting index name to {settings.index_name} and image directory to {settings.image_dir}")
    INDEX_NAME = settings.index_name
    IMAGE_DIR = settings.image_dir
    logger.info(f"Settings updated: index_name={INDEX_NAME}, image_dir={IMAGE_DIR}")
    return {"message": "Settings updated successfully", "index_name": INDEX_NAME, "image_dir": IMAGE_DIR}

# --------------------
# Image Operations
# --------------------
@image_router.post("/search", response_model=ImageSearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    logger.info(f"Received request to search for similar images using file: {file.filename}.")
    try:
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            image.save(tmp, format="JPEG")
            tmp.seek(0)
            image_vector = get_image_embedding(tmp.name)
        response = search_knn(image_vector, INDEX_NAME, num_images=num_images)
        image_ids = [hit['_source']['image_id'] for hit in response['hits']['hits']]
        query_image_id = file.filename
        return ImageSearchResult(query_image_id=query_image_id, similar_image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_image: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {e}")

@image_router.post("/batch-index")
async def batch_index_images(index_name: str = Query(...), image_dir: str = Query(...)):
    logger.info(f"Received batch index request for index: {index_name}, image directory: {image_dir}")
    try:
        create_index(client, index_name)
        dataloader = create_dataloader(image_dir, batch_size=32, num_workers=4)
        embeddings = generate_embeddings(dataloader)
        response = bulk_index_embeddings(client, index_name, embeddings)
        return {"message": "Batch indexing completed successfully", "details": response}
    except Exception as e:
        logger.error(f"Error during batch indexing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during batch indexing: {e}")

# --------------------
# Text Operations
# --------------------
@text_router.post("/search", response_model=SearchResult)
async def search_text(request: TextQueryRequest):
    logger.info(f"Received text search request: {request.json()}")
    try:
        vector = get_text_embedding(request.query)
        response = search_knn(vector, INDEX_NAME, request.num_images)
        image_ids = [hit['_source']['image_id'] for hit in response['hits']['hits']]
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@text_router.post("/batch-index")
async def batch_index_text(index_name: str = Query(...), text_file_path: str = Query(...)):
    logger.info(f"Received batch index request for text index: {index_name}, text file: {text_file_path}")
    try:
        create_index_text(index_name, os_client=client, dimension=512)
        dataloader = create_text_dataloader(text_file_path, batch_size=32, num_workers=4)
        embeddings = generate_text_embeddings(dataloader)
        response = bulk_index_text_embeddings(client, index_name, embeddings)
        return {"message": "Batch indexing completed successfully", "details": response}
    except Exception as e:
        logger.error(f"Error during batch text indexing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --------------------
# Face Operations
# --------------------
@face_router.post("/single", response_model=FaceDetectionResult)
async def process_single_image(file: UploadFile = File(...), index_name: Optional[str] = Query(None)):
    logger.info(f"Received request to process a single face image with index_name={index_name}.")
    try:
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        detected_faces = detect_faces_in_image(image)
        embeddings = generate_face_embeddings(image, detected_faces, index_name)
        return {"message": "Face processed successfully", "embeddings": embeddings}
    except Exception as e:
        logger.error(f"Error processing single face image: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@face_router.post("/batch")
async def process_faces_and_index(image_dir: str = Query(...), face_crop_dir: str = Query(...), index_name: str = Query(...), batch_size: int = Query(32)):
    logger.info(f"Received request to process faces and index for directory {image_dir}.")
    try:
        create_index(client, index_name)
        if not os.path.exists(face_crop_dir):
            os.makedirs(face_crop_dir)
        for image_file in os.listdir(image_dir):
            detect_and_crop_faces(os.path.join(image_dir, image_file), face_crop_dir)
        dataset = CroppedFaceDataset(face_crop_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        response = generate_embeddings_and_index(client, face_crop_dir, index_name, dataloader)
        return {"message": "Faces processed and indexed successfully", "details": response}
    except Exception as e:
        logger.error(f"Error during face processing and indexing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --------------------
# Root Endpoint
# --------------------
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Search API"}

# Include routers
app.include_router(image_router)
app.include_router(text_router)
app.include_router(face_router)