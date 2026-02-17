# ============================================================
# FastAPI AWS Lightweight Entry Point
# Only includes Image Search + Text Search (no Face, no metrics)
# Uses LAZY loading — heavy imports (torch, clip) are deferred
# ============================================================

import io
import os
import logging
import tempfile
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, APIRouter

# Data models — lightweight, no heavy deps
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

# ---- Lazy-loaded module cache ----
_modules = {}


def _get_opensearch_client():
    """Lazy-load the OpenSearch client."""
    if "client" not in _modules:
        logger.info("Initializing OpenSearch client...")
        from dev.image_embedding.open_search_client import client
        _modules["client"] = client
        logger.info("OpenSearch client ready.")
    return _modules["client"]


def _get_clip_functions():
    """Lazy-load CLIP model and related functions (triggers torch import)."""
    if "clip_loaded" not in _modules:
        logger.info("Loading CLIP model (this may take a few minutes on first call)...")
        from dev.image_embedding.embedding_generation import get_image_embedding, get_text_embedding
        from dev.image_embedding.search import search_knn
        from dev.image_embedding.create_image_index import create_index
        _modules["get_image_embedding"] = get_image_embedding
        _modules["get_text_embedding"] = get_text_embedding
        _modules["search_knn"] = search_knn
        _modules["create_index"] = create_index
        _modules["clip_loaded"] = True
        logger.info("CLIP model loaded successfully!")
    return _modules


def _get_text_functions():
    """Lazy-load text embedding functions."""
    if "text_loaded" not in _modules:
        logger.info("Loading text embedding modules...")
        from dev.text_embedding.create_text_index import create_index_text
        from dev.text_embedding.text_batch_embedding import (
            bulk_index_text_embeddings,
            create_text_dataloader,
            generate_text_embeddings,
        )
        _modules["create_index_text"] = create_index_text
        _modules["bulk_index_text_embeddings"] = bulk_index_text_embeddings
        _modules["create_text_dataloader"] = create_text_dataloader
        _modules["generate_text_embeddings"] = generate_text_embeddings
        _modules["text_loaded"] = True
        logger.info("Text embedding modules loaded.")
    return _modules


# Routers
image_router = APIRouter(prefix="/images", tags=["Image Operations"])
text_router = APIRouter(prefix="/texts", tags=["Text Operations"])


# =====================
# Root & Settings (instant — no heavy imports)
# =====================
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Search API"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        client = _get_opensearch_client()
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
# Image Operations (lazy-loads CLIP on first call)
# =====================
@image_router.post("/search", response_model=ImageSearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    """Search for similar images using CLIP embeddings."""
    logger.info(f"Image search request: {file.filename}")
    m = _get_clip_functions()
    try:
        from PIL import Image
        file_content = await file.read()
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp, format="JPEG")
            tmp_path = tmp.name

        image_vector = m["get_image_embedding"](tmp_path)
        os.unlink(tmp_path)
        logger.info(f"Embedding generated for {file.filename}")

        response = m["search_knn"](image_vector, INDEX_NAME, num_images=num_images)

        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return ImageSearchResult(query_image_id=file.filename, similar_image_ids=image_ids)

    except Exception as e:
        logger.error(f"Error during search_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@image_router.post("/batch-index")
async def batch_index_images(index_name: str = Query(...), image_dir: str = Query(...)):
    """Batch index images into OpenSearch — memory-efficient chunked version."""
    logger.info(f"Batch index request: index={index_name}, dir={image_dir}")
    client = _get_opensearch_client()

    # Lazy import only what we need (avoid image_batch_embedding.py which loads CLIP again)
    import torch
    import gc
    from PIL import Image
    from opensearchpy.helpers import bulk

    # Load CLIP via our lazy loader (reuses already-loaded model)
    m = _get_clip_functions()

    # Create the index
    m["create_index"](client, index_name)

    # Get the CLIP model and preprocess from embedding_generation
    from dev.image_embedding.embedding_generation import model, preprocess, device

    # List all images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)
    logger.info(f"Found {total} images to index")

    CHUNK_SIZE = 50  # Process 50 images at a time
    indexed = 0
    errors = 0

    for chunk_start in range(0, total, CHUNK_SIZE):
        chunk_files = image_files[chunk_start:chunk_start + CHUNK_SIZE]
        actions = []

        for fname in chunk_files:
            try:
                img_path = os.path.join(image_dir, fname)
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding /= embedding.norm(dim=-1, keepdim=True)

                image_id = os.path.splitext(fname)[0]
                actions.append({
                    "_index": index_name,
                    "_id": chunk_start + len(actions),
                    "_source": {
                        "my_vector": embedding.squeeze().cpu().tolist(),
                        "image_id": image_id,
                    }
                })
                # Free tensor memory
                del image_tensor, embedding, image
            except Exception as e:
                errors += 1
                logger.warning(f"Skipping {fname}: {e}")

        # Index this chunk to OpenSearch
        if actions:
            try:
                bulk(client, actions)
                indexed += len(actions)
                logger.info(f"Indexed {indexed}/{total} images ({errors} errors)")
            except Exception as e:
                logger.error(f"Bulk index failed for chunk at {chunk_start}: {e}")

        # Free memory
        del actions
        gc.collect()

    logger.info(f"Batch indexing complete: {indexed}/{total} indexed, {errors} errors")
    return {
        "message": "Batch indexing completed",
        "total_images": total,
        "indexed": indexed,
        "errors": errors,
    }


# =====================
# Text Operations (lazy-loads on first call)
# =====================
@text_router.post("/search", response_model=SearchResult)
async def search_text(request: TextQueryRequest):
    """Search for images using a text query via CLIP text encoder."""
    logger.info(f"Text search request: '{request.query}'")
    m = _get_clip_functions()
    try:
        vector = m["get_text_embedding"](request.query)
        response = m["search_knn"](vector, INDEX_NAME, request.num_images)
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@text_router.post("/batch-index")
async def batch_index_text(index_name: str = Query(...), text_file_path: str = Query(...)):
    """Batch index text metadata into OpenSearch."""
    logger.info(f"Text batch index: index={index_name}, file={text_file_path}")
    m = _get_text_functions()
    client = _get_opensearch_client()
    try:
        m["create_index_text"](index_name, os_client=client, dimension=512)
        dataloader = m["create_text_dataloader"](text_file_path, batch_size=16, num_workers=2)
        embeddings = m["generate_text_embeddings"](dataloader)
        response = m["bulk_index_text_embeddings"](client, index_name, embeddings)
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
