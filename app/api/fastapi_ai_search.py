# Standard library imports
import argparse
import glob
import io
import logging
import os
import uuid
import tempfile
from typing import List, Optional
import asyncio
from typing import Dict, Any, Optional


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
from dev.face_embedding.face_search import perform_knn_search
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
    "*"
    # "http://localhost",
    # "http://localhost:3000",
    # "http://localhost:8080",
    # # Add more origins as needed
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

class ProcessedImage(BaseModel):
    bounding_box: List[int]
    confidence: float
    cropped_width: int
    cropped_height: int
    cropped_resolution: str
    cropped_image_path: str

class FaceEmbedding(BaseModel):
    # Adjust the fields based on your actual embedding structure
    face_id: str
    box: List[int]
    embedding: List[float] 

class FaceDetectionResult(BaseModel):
    status: str
    image_name: str
    total_faces_detected: int
    processed_images: List[ProcessedImage]
    embeddings: List[FaceEmbedding]

class IndexingDetail(BaseModel):
    image_name: str
    face_id: Optional[str]
    status: str  # e.g., "indexed" or "error"
    error_message: Optional[str] = None

class BatchEmbeddingDetails(BaseModel):
    total_images_processed: int
    total_faces_detected: int
    total_embeddings_indexed: int
    indexing_details: List[IndexingDetail]

class BatchFaceEmbeddingResult(BaseModel):
    status: str
    face_crop_dir: str
    details: BatchEmbeddingDetails

class CombinedSearchDetail(BaseModel):
    image_search: Optional[ImageSearchResult]
    face_search: Optional[Dict[str, Any]]

class CombinedSearchResponse(BaseModel):
    status: str
    details: CombinedSearchDetail
    error: Optional[str] = None

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

# Subsection 1: Detection
@face_router.post("/detection/single")
async def detect_faces_in_single_image(file: UploadFile = File(...)):
    logger.info(f"Received request to detect faces in a single image: {file.filename}")
    try:
        # Open the image file
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get original image dimensions
        original_width, original_height = image.size

        # Detect faces in the image
        try:
            detected_faces = detect_faces_in_image(image)  # This returns bounding boxes and probabilities
        except ValueError as e:
            return {
                "status": "No faces detected",
                "image_name": file.filename,
                "image_resolution": f"{original_width}x{original_height}",
                "processed_images": []
            }
        # Get the number of faces detected
        total_faces = len(detected_faces)

        # Prepare JSON response
        processed_images = []
        for index, face in enumerate(detected_faces):
            # Extract bounding box and confidence
            x_min, y_min, x_max, y_max = face["box"]
            confidence = face["probability"]

            # Crop the face
            cropped_face = image.crop((x_min, y_min, x_max, y_max))
            cropped_width, cropped_height = cropped_face.size

            # Save the cropped face
            save_dir = "cropped_faces"  # Directory to save cropped faces
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cropped_filename = f"{file.filename.split('.')[0]}_face_{index + 1}.jpg"
            cropped_face_path = os.path.join(save_dir, cropped_filename)
            cropped_face.save(cropped_face_path)

            # Append details
            processed_images.append({
                "bounding_box": [x_min, y_min, x_max, y_max],
                "confidence": round(confidence, 2),
                "cropped_width": cropped_width,
                "cropped_height": cropped_height,
                "cropped_resolution": f"{cropped_width}x{cropped_height}"
            })

        return {
            "status": "Faces detected",
            "image_name": file.filename,
            "image_resolution": f"{original_width}x{original_height}",
            "total_faces_detected": total_faces,
            "processed_images": processed_images
        }

    except Exception as e:
        logger.error(f"Error detecting faces in single image: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@face_router.post("/search")
async def search_similar_faces(
    file: UploadFile = File(...),
    index_name: str = Query(...),
    k: int = Query(5)
):
    """
    Search for similar faces in the index using the uploaded image.
    
    Args:
        file (UploadFile): The uploaded image file.
        index_name (str): The name of the OpenSearch index.
        k (int): The number of nearest neighbors to retrieve.

    Returns:
        JSON response with search results.
    """
    try:
        # Open the uploaded image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Detect faces in the image
        detected_faces = detect_faces_in_image(image)
        if not detected_faces:
            return {
                "status": "No faces detected",
                "image_name": file.filename,
                "processed_images": []
            }

        # Prepare the search results
        search_results = []
        for index, face in enumerate(detected_faces):
            # Extract bounding box
            x_min, y_min, x_max, y_max = face["box"]

            # Crop the face
            cropped_face = image.crop((x_min, y_min, x_max, y_max))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cropped_face.save(tmp, format="JPEG")
                tmp.seek(0)
                cropped_face_path = tmp.name

            # Generate embedding for the cropped face
            embedding = get_image_embedding(cropped_face_path)

            # Perform the face similarity search
            response = search_knn(embedding, index_name, k)
            similar_faces = [hit['_source'] for hit in response['hits']['hits']]

            search_results.append({
                "face_id": str(uuid.uuid4()),
                "box": face["box"],
                "similar_faces": similar_faces
            })

        return {
            "status": "Search completed",
            "image_name": file.filename,
            "total_faces_detected": len(detected_faces),
            "search_results": search_results
        }

    except Exception as e:
        logger.error(f"Error during face search: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
@face_router.post("/detection/batch")
async def detect_faces_batch(image_dir: str = Query(...), face_crop_dir: str = Query(...)):
    """
    Detect faces in a batch of images and return details for each image.
    """
    logger.info(f"Received batch face detection request for directory: {image_dir}")
    try:
        # Ensure the output directory for cropped faces exists
        if not os.path.exists(face_crop_dir):
            os.makedirs(face_crop_dir)

        # Initialize response list
        batch_results = []

        margin = 25  # Margin to add to face crops
        # Process each image in the input directory
        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            try:
                # Load and preprocess the image
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Detect faces in the image
                detected_faces = detect_faces_in_image(image)
                total_faces = len(detected_faces)

                # Process detected faces and save crops
                processed_faces = []
                for face in detected_faces:
                    x1, y1, x2, y2 = face["box"]
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(image.width, x2 + margin)
                    y2 = min(image.height, y2 + margin)
                    
                    cropped_width = x2 - x1
                    cropped_height = y2 - y1
                    cropped_resolution = f"{cropped_width}x{cropped_height}"

                    # Save the cropped face to the specified directory
                    face_crop = image.crop((x1, y1, x2, y2))
                    crop_filename = f"{os.path.splitext(image_file)[0]}_face_{uuid.uuid4().hex}.jpg"
                    crop_path = os.path.join(face_crop_dir, crop_filename)
                    face_crop.save(crop_path)

                    # Append processed face details
                    processed_faces.append({
                        "bounding_box": [x1, y1, x2, y2],
                        "confidence": face["probability"],
                        "cropped_width": cropped_width,
                        "cropped_height": cropped_height,
                        "cropped_resolution": cropped_resolution,
                        "cropped_image_path": crop_path
                    })

                # Append results for this image
                batch_results.append({
                    "image_name": image_file,
                    "image_resolution": f"{image.width}x{image.height}",
                    "total_faces_detected": total_faces,
                    "processed_faces": processed_faces
                })
            except Exception as image_error:
                logger.error(f"Error processing image {image_file}: {image_error}")
                batch_results.append({
                    "image_name": image_file,
                    "error": str(image_error)
                })

        return {
            "status": "Batch processing completed",
            "total_images_processed": len(batch_results),
            "results": batch_results
        }

    except Exception as e:
        logger.error(f"Error during batch face detection: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



# Subsection 2: Embedding
@face_router.post("/embedding/single", response_model=FaceDetectionResult)
async def embed_faces_single_image(file: UploadFile = File(...), index_name: Optional[str] = Query(None)):
    """
    Generate embeddings for a single face image and optionally index it.
    """
    logger.info(f"Received request to generate embeddings for a single image with index_name={index_name}.")
    try:
        # Open the image
        image = Image.open(file.file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Detect faces in the image
        detected_faces = detect_faces_in_image(image)

        if not detected_faces:
            return {
                "status": "No faces detected",
                "image_name": file.filename,
                "processed_images": []
            }

        # Generate embeddings for the detected faces
        embeddings = generate_face_embeddings(image, detected_faces)

        # Prepare embeddings list matching FaceEmbedding model
        embeddings_list = []
        for index, embedding_dict in enumerate(embeddings):
            embedding = embedding_dict['embedding']
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()  # Convert to list if it's a NumPy array

            embeddings_list.append({
                "face_id": embedding_dict.get('face_id', str(uuid.uuid4())),
                "box": embedding_dict.get('box', detected_faces[index]["box"]),
                "embedding": embedding
            })

        # Prepare processed_images list
        processed_images = []
        for index, face in enumerate(detected_faces):
            # Extract bounding box and confidence
            x_min, y_min, x_max, y_max = face["box"]
            confidence = face["probability"]

            # Crop the face
            cropped_face = image.crop((x_min, y_min, x_max, y_max))
            cropped_width, cropped_height = cropped_face.size

            # Save the cropped face
            save_dir = "cropped_faces"  # Directory to save cropped faces
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cropped_filename = f"{file.filename.split('.')[0]}_face_{index + 1}.jpg"
            cropped_face_path = os.path.join(save_dir, cropped_filename)
            cropped_face.save(cropped_face_path)

            # Append details
            processed_images.append({
                "bounding_box": [x_min, y_min, x_max, y_max],
                "confidence": round(confidence, 2),
                "cropped_width": cropped_width,
                "cropped_height": cropped_height,
                "cropped_resolution": f"{cropped_width}x{cropped_height}",
                "cropped_image_path": cropped_face_path
            })

        # Prepare and return the response
        return {
            "status": "Face embeddings generated successfully",
            "image_name": file.filename,
            "total_faces_detected": len(detected_faces),
            "processed_images": processed_images,
            "embeddings": embeddings_list
        }
    except Exception as e:
        logger.error(f"Error generating embeddings for single face image: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@face_router.post("/embedding/batch", response_model=BatchFaceEmbeddingResult)
async def embed_faces_batch(
    image_dir: str = Query(...),
    face_crop_dir: str = Query(...),
    index_name: str = Query(...),
    batch_size: int = Query(32)
):
    logger.info(f"Received batch embedding request for directory: {image_dir}")
    try:
        create_index(client, index_name)  # Ensure the index exists

        # Create directory for cropped faces if it doesn't exist
        os.makedirs(face_crop_dir, exist_ok=True)

        # Initialize counters and details list
        total_images_processed = 0
        total_faces_detected = 0
        indexing_details = []

        # Detect faces and crop them for each image in the directory
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            try:
                faces_detected = detect_and_crop_faces(image_path, face_crop_dir)
                total_images_processed += 1
                total_faces_detected += len(faces_detected)  # Assuming faces_detected is an integer
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")
                indexing_details.append(IndexingDetail(
                    image_name=image_file,
                    face_id=None,
                    status="error",
                    error_message=str(e)
                ))
                continue

        # Process the cropped faces and generate embeddings
        dataset = CroppedFaceDataset(face_crop_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Generate embeddings and index them
        total_embeddings_indexed, indexing_failures = generate_embeddings_and_index(
            client, face_crop_dir, index_name, dataloader
        )

        # Append any indexing failures to indexing_details
        for item in indexing_failures:
            indexing_details.append(IndexingDetail(
                image_name=item.get('image_name', ''),
                face_id=item.get('face_id', ''),
                status="error",
                error_message=item.get('error_message', '')
            ))

        # Prepare the details object
        details = BatchEmbeddingDetails(
            total_images_processed=total_images_processed,
            total_faces_detected=total_faces_detected,
            total_embeddings_indexed=total_embeddings_indexed,
            indexing_details=indexing_details
        )

        # Return the response
        return BatchFaceEmbeddingResult(
            status="Batch face embedding and indexing completed successfully",
            face_crop_dir=face_crop_dir,
            details=details
        )
    except Exception as e:
        logger.error(f"Error during batch face embedding and indexing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --------------------
# Combined Operations
# --------------------

# Create combined router
combined_router = APIRouter(prefix="/combined", tags=["Combined Operations"])

@combined_router.post("/search", response_model=CombinedSearchResponse)
async def combined_search(
    file: UploadFile = File(...),
    image_index: str = Query(..., description="Index name for general image search"),
    face_index: str = Query(..., description="Index name for face search"),
    num_images: int = Query(5, description="Number of results to return for both searches")
):
    """Performs parallel image and face search using separate indexes"""
    logger.info(f"Starting combined search for file: {file.filename}")
    try:
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp, format="JPEG")
            tmp_path = tmp.name

        async def do_image_search():
            try:
                image_vector = get_image_embedding(tmp_path)
                response = search_knn(image_vector, image_index, num_images)  # Use image_index
                return ImageSearchResult(
                    query_image_id=file.filename,
                    similar_image_ids=[hit['_source']['image_id'] for hit in response['hits']['hits']]
                )
            except Exception as e:
                logger.error(f"Image search error: {e}")
                return None

        async def do_face_search():
            try:
                faces = detect_faces_in_image(image)
                if not faces:
                    return {"status": "No faces detected"}

                results = []
                for face in faces:
                    box = face["box"]
                    face_crop = image.crop(box)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as face_tmp:
                        face_crop.save(face_tmp, format="JPEG")
                        embedding = get_image_embedding(face_tmp.name)
                        os.unlink(face_tmp.name)
                    
                    matches = search_knn(embedding, face_index, num_images)  # Use face_index
                    results.append({
                        "face_id": str(uuid.uuid4()),
                        "box": box,
                        "matches": [hit['_source'] for hit in matches['hits']['hits']]
                    })
                
                return {
                    "status": "success",
                    "faces_found": len(faces),
                    "results": results
                }
            except Exception as e:
                logger.error(f"Face search error: {e}")
                return None

        # Execute searches in parallel
        image_results, face_results = await asyncio.gather(
            do_image_search(),
            do_face_search(),
            return_exceptions=True
        )

        return CombinedSearchResponse(
            status="success",
            details=CombinedSearchDetail(
                image_search=image_results if not isinstance(image_results, Exception) else None,
                face_search=face_results if not isinstance(face_results, Exception) else None
            )
        )

    except Exception as e:
        logger.error(f"Combined search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

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
app.include_router(combined_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)