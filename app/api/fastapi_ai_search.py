# Standard library imports
import argparse
import glob
import io
import os
import tempfile
from typing import List, Optional


# Third-party library imports
import PIL
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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
app = FastAPI()

# Index and image directory configuration
INDEX_NAME = ''
IMAGE_DIR = ""


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
    INDEX_NAME = settings.index_name
    IMAGE_DIR = settings.image_dir
    return {"message": "Settings updated successfully", "index_name": INDEX_NAME, "image_dir": IMAGE_DIR}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Search API. Use POST /search-text to search images."}

@app.post("/search_text_single", response_model=SearchResult)
async def search_text(request: TextQueryRequest):
    print(f"Received search request: {request.json()}")

    try:
        # Generate the text embedding using CLIP
        vector = get_text_embedding(request.query)
        
        # Perform KNN search on OpenSearch
        response = search_knn(vector, INDEX_NAME, request.num_images)
        
        # Extract image IDs from search results
        image_ids = [hit['_source']['image_id'] for hit in response['hits']['hits']]
        
        # Return the image IDs as response
        return SearchResult(image_ids=image_ids)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/search_image_single", response_model=ImageSearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    try:
        # Read the uploaded image file
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Generate the image embedding using CLIP
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            image.save(tmp, format="JPEG")
            tmp.seek(0)
            image_vector = get_image_embedding(tmp.name)

        # Perform KNN search on OpenSearch
        response = search_knn(image_vector, INDEX_NAME, num_images=num_images)
        
        # Extract image IDs from search results
        image_ids = [hit['_source']['image_id'] for hit in response['hits']['hits']]

        # Return the image IDs along with the query image ID
        query_image_id = file.filename
        return ImageSearchResult(query_image_id=query_image_id, similar_image_ids=image_ids)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {e}")


@app.post("/batch-index-images")
async def batch_index_images(index_name: str = Query(...), image_dir: str = Query(...)):
    try:
        # Create the index in OpenSearch
        create_index(client, index_name)

        # Create a DataLoader for the images using the provided image directory
        dataloader = create_dataloader(image_dir, batch_size=32, num_workers=4)
        print("length of dataloader: ", len(dataloader))
        # Generate embeddings for all images in the DataLoader
        embeddings = generate_embeddings(dataloader)
        print("length of embeddings: ", len(embeddings))
        # Index the embeddings in OpenSearch using the provided index name
        response = bulk_index_embeddings(client, index_name, embeddings)
        print("response: ", response)
        return {"message": "Batch indexing completed successfully", "details": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during batch indexing: {e}")
    

# @app.post("/batch-index-text")
# async def batch_index_text(index_name: str = Query(...), text_file_path: str = Query(...)):
#     try:
#         # Create the index in OpenSearch (function `create_index` should be defined to create an index)
#         create_index(client, index_name)

#         # Create a DataLoader for the text data
#         dataloader = create_text_dataloader(text_file_path, batch_size=32, num_workers=4)
#         print("Length of dataloader: ", len(dataloader))

#         # Generate embeddings for all text data in the DataLoader
#         embeddings = generate_text_embeddings(dataloader)
#         print("Length of embeddings: ", len(embeddings))

#         # Index the embeddings in OpenSearch using the provided index name
#         response = bulk_index_text_embeddings(client, index_name, embeddings)
#         print("Response: ", response)

#         return {"message": "Batch indexing completed successfully", "details": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred during batch indexing: {e}")

@app.post("/batch-index-text")
async def batch_index_text(index_name: str = Query(...), text_file_path: str = Query(...)):
    try:
        #  Create the index in OpenSearch (function `create_index` should be defined to create an index)
        create_index_text(index_name, os_client=client,  dimension=512)

        # Create a dataloader from the provided text file
        dataloader = create_text_dataloader(text_file_path, batch_size=32, num_workers=4)
        
        # Generate embeddings using the dataloader
        embeddings = generate_text_embeddings(dataloader)
        
        # Index the embeddings in OpenSearch (not shown in the snippet, but this would use `bulk_index_text_embeddings`)
        response = bulk_index_text_embeddings(client, index_name, embeddings)
        
        return {"message": "Batch indexing completed successfully", "details": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during batch indexing: {e}")
    
@app.post("/process-single-face-image")
async def process_single_image(
    file: UploadFile = File(...),
    index_name: Optional[str] = Query(None)
):
    """
    Process a single image: Detect faces, generate embeddings, and optionally index them in OpenSearch.
    """
    try:
        # Load the image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Detect faces
        detected_faces = detect_faces_in_image(image)

        # Generate embeddings
        embeddings = generate_face_embeddings(image, detected_faces, index_name)

        return {"message": "Face processed successfully", "embeddings": embeddings}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



# Endpoint for face detection and embedding generation
@app.post("/process-faces-and-index")
async def process_faces_and_index(image_dir: str = Query(...), face_crop_dir: str = Query(...), index_name: str = Query(...), batch_size: int = Query(32)):
    """
    Detects faces, generates embeddings in batches using DataLoader, and indexes them in OpenSearch.
    """
    try:
        create_index(client, index_name)
        # Step 1: Validate directories
        if not os.path.exists(image_dir):
            raise HTTPException(status_code=400, detail=f"Image directory '{image_dir}' does not exist.")
        if not os.path.exists(face_crop_dir):
            os.makedirs(face_crop_dir)

        # Step 2: Detect and crop faces (using previous function)
        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            detect_and_crop_faces(image_path, face_crop_dir)

        # Step 3: Initialize DataLoader
        dataset = CroppedFaceDataset(face_crop_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Step 4: Generate embeddings and index them
        response = generate_embeddings_and_index(client, face_crop_dir, index_name, dataloader)

        return {"message": "Faces processed and indexed successfully", "details": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


