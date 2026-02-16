# ============================================================
# FastAPI AWS Lightweight Entry Point
# Only includes Image Search + Text Search (no Face, no metrics)
# ============================================================

# Standard library imports
import io
import os
import logging
import tempfile
from typing import List, Optional

# Third-party library imports
from PIL import Image
from torch.utils.data import DataLoader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, APIRouter

# Local application imports — IMAGE & TEXT ONLY (no face imports)
from dev.image_embedding.create_image_index import create_index
from dev.image_embedding.embedding_generation import get_image_embedding, get_text_embedding
from dev.image_embedding.image_batch_embedding import (
    create_dataloader,
    generate_embeddings,
    bulk_index_embeddings,
)
from dev.image_embedding.open_search_client import client
from dev.image_embedding.search import search_knn
from dev.text_embedding.create_text_index import create_index_text
from dev.text_embedding.text_batch_embedding import (
    bulk_index_text_embeddings,
    create_text_dataloader,
    generate_text_embeddings,
)

# Data models — only the ones we need
from app.model.data_model import (
    TextQueryRequest,
    SearchResult,
    ImageSearchResult,
    SettingsRequest,
)

# ---- App Setup ----
app = FastAPI(
    title="AI Search API",
    description="Multimodal image & text search powered by CLIP + OpenSearch",
    version="1.0.0",
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# CORS — allow all for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
INDEX_NAME = ""
IMAGE_DIR = ""

# Routers
image_router = APIRouter(prefix="/images", tags=["Image Operations"])
text_router = APIRouter(prefix="/texts", tags=["Text Operations"])


# =====================
# Root & Settings
# =====================
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Search API"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check OpenSearch connectivity
        info = client.info()
        os_status = "connected"
    except Exception:
        os_status = "disconnected"

    return {
        "status": "healthy",
        "opensearch": os_status,
        "version": "1.0.0",
    }


@app.post("/set-settings")
async def set_settings(settings: SettingsRequest):
    global INDEX_NAME, IMAGE_DIR
    logger.info(f"Setting index_name={settings.index_name}, image_dir={settings.image_dir}")
    INDEX_NAME = settings.index_name
    IMAGE_DIR = settings.image_dir
    return {
        "message": "Settings updated successfully",
        "index_name": INDEX_NAME,
        "image_dir": IMAGE_DIR,
    }


# =====================
# Image Operations
# =====================
@image_router.post("/search", response_model=ImageSearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    """Search for similar images using CLIP embeddings."""
    logger.info(f"Image search request: {file.filename}")
    try:
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp, format="JPEG")
            tmp_path = tmp.name

        image_vector = get_image_embedding(tmp_path)
        os.unlink(tmp_path)
        logger.info(f"Embedding generated for {file.filename}")

        response = search_knn(image_vector, INDEX_NAME, num_images=num_images)

        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return ImageSearchResult(query_image_id=file.filename, similar_image_ids=image_ids)

    except Exception as e:
        logger.error(f"Error during search_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@image_router.post("/batch-index")
async def batch_index_images(index_name: str = Query(...), image_dir: str = Query(...)):
    """Batch index images into OpenSearch."""
    logger.info(f"Batch index request: index={index_name}, dir={image_dir}")
    try:
        create_index(client, index_name)
        dataloader = create_dataloader(image_dir, batch_size=16, num_workers=2)
        embeddings = generate_embeddings(dataloader)
        response = bulk_index_embeddings(client, index_name, embeddings)
        return {"message": "Batch indexing completed successfully", "details": response}
    except Exception as e:
        logger.error(f"Error during batch indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# =====================
# Text Operations
# =====================
@text_router.post("/search", response_model=SearchResult)
async def search_text(request: TextQueryRequest):
    """Search for images using a text query via CLIP text encoder."""
    logger.info(f"Text search request: '{request.query}'")
    try:
        vector = get_text_embedding(request.query)
        response = search_knn(vector, INDEX_NAME, request.num_images)
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@text_router.post("/batch-index")
async def batch_index_text(index_name: str = Query(...), text_file_path: str = Query(...)):
    """Batch index text metadata into OpenSearch."""
    logger.info(f"Text batch index: index={index_name}, file={text_file_path}")
    try:
        create_index_text(index_name, os_client=client, dimension=512)
        dataloader = create_text_dataloader(text_file_path, batch_size=16, num_workers=2)
        embeddings = generate_text_embeddings(dataloader)
        response = bulk_index_text_embeddings(client, index_name, embeddings)
        return {"message": "Batch indexing completed successfully", "details": response}
    except Exception as e:
        logger.error(f"Error during batch text indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# ---- Include Routers ----
app.include_router(image_router)
app.include_router(text_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
