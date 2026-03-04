"""
FastAPI backend for Hugging Face Spaces deployment.
Replaces OpenSearch with FAISS for zero-cost vector search.
Uses ONNX Runtime for CLIP inference (same as AWS version).
"""

import os
import sys
import time
import json
import logging
from typing import List, Optional

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
APP_BASE_DIR = os.getenv("APP_BASE_DIR", "/app")
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", os.path.join(APP_BASE_DIR, "model", "clip_onnx"))
CLIP_MODEL_DIR = os.getenv("CLIP_MODEL_DIR", os.path.join(APP_BASE_DIR, "model", "clip"))
FRONTEND_DIR = os.getenv("FRONTEND_DIR", os.path.join(APP_BASE_DIR, "frontend"))
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", os.path.join(APP_BASE_DIR, "datalake", "embeddings.jsonl"))

# Initialize FastAPI
app = FastAPI(title="AI Search API (HF Spaces Edition)", version="3.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
image_router = APIRouter(prefix="/images", tags=["Image Operations"])
text_router = APIRouter(prefix="/texts", tags=["Text Operations"])
admin_router = APIRouter(prefix="/admin", tags=["Admin Operations"])

# --- Models ---
class SearchResult(BaseModel):
    image_ids: List[str]

class TextSearchRequest(BaseModel):
    query: str
    num_images: int = 5

# --- Globals ---
_inference_engine = None
_onnx_text_session = None
_onnx_image_session = None
_clip_model = None
_clip_preprocess = None
_clip_tokenize = None

# FAISS index and ID mapping
_faiss_index = None
_image_ids = []  # Maps FAISS index position -> image_id string


# ==================================================
# FAISS Index (replaces OpenSearch)
# ==================================================

def _build_faiss_index():
    """Load embeddings.jsonl and build a FAISS index for cosine similarity search."""
    global _faiss_index, _image_ids

    if _faiss_index is not None:
        return  # Already built

    logger.info(f"Building FAISS index from {EMBEDDINGS_PATH}...")
    t0 = time.time()

    embeddings = []
    image_ids = []

    with open(EMBEDDINGS_PATH, 'r') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            image_ids.append(data["image_id"])
            embeddings.append(data["embedding"])

            if (line_num + 1) % 5000 == 0:
                logger.info(f"  Loaded {line_num + 1} embeddings...")

    # Convert to numpy array
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize vectors (for cosine similarity via inner product)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors = vectors / norms

    # Build FAISS index (IndexFlatIP = inner product on normalized vectors = cosine similarity)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    _faiss_index = index
    _image_ids = image_ids

    elapsed = time.time() - t0
    logger.info(f"FAISS index built: {len(image_ids)} vectors, {dimension}-dim, {elapsed:.1f}s")
    logger.info(f"   Memory: ~{vectors.nbytes / 1024 / 1024:.1f}MB for vectors")


def search_faiss(vector, k=5):
    """Search FAISS index for k nearest neighbors."""
    if _faiss_index is None:
        raise RuntimeError("FAISS index not built yet")

    # Prepare query vector
    query = np.array([vector], dtype=np.float32)
    # Normalize query vector
    norm = np.linalg.norm(query, axis=1, keepdims=True)
    if norm[0][0] > 0:
        query = query / norm

    # Search
    distances, indices = _faiss_index.search(query, k)

    # Map indices to image IDs
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_image_ids):
            results.append(_image_ids[idx])

    return results


# ==================================================
# CLIP Inference (same as AWS version)
# ==================================================

def _get_clip_preprocess():
    """Create CLIP ViT-B/32 preprocessing transform WITHOUT loading the model."""
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        interp = InterpolationMode.BICUBIC
    except ImportError:
        from PIL import Image as PILImage
        interp = PILImage.BICUBIC

    return Compose([
        Resize(224, interpolation=interp),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])


def _init_inference():
    """Initialize the inference engine: ONNX if available, else PyTorch."""
    global _inference_engine, _onnx_text_session, _onnx_image_session
    global _clip_model, _clip_preprocess, _clip_tokenize

    if _inference_engine is not None:
        return

    text_onnx = os.path.join(ONNX_MODEL_DIR, "clip_text_encoder.onnx")
    image_onnx = os.path.join(ONNX_MODEL_DIR, "clip_image_encoder.onnx")

    # --- Lazy ONNX Export (runs once on first startup) ---
    if not os.path.exists(text_onnx) or not os.path.exists(image_onnx):
        logger.info("⏳ ONNX models not found. Exporting from PyTorch (one-time operation)...")
        try:
            import subprocess
            os.makedirs(ONNX_MODEL_DIR, exist_ok=True)
            result = subprocess.run(
                ["python", "scripts/export_clip_onnx.py",
                 "--output-dir", ONNX_MODEL_DIR,
                 "--model-cache", CLIP_MODEL_DIR],
                capture_output=True, text=True, timeout=1800
            )
            if result.returncode == 0:
                logger.info("✅ ONNX export complete!")
            else:
                logger.error(f"❌ ONNX export failed: {result.stderr}")
        except Exception as e:
            logger.error(f"❌ ONNX export error: {e}")

    if os.path.exists(text_onnx) and os.path.exists(image_onnx):
        logger.info("🚀 ONNX models found! Using ONNX Runtime for inference.")
        import onnxruntime as ort
        import clip

        t0 = time.time()
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 4
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _onnx_text_session = ort.InferenceSession(
            text_onnx,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )
        _onnx_image_session = ort.InferenceSession(
            image_onnx,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )
        logger.info(f"  ONNX sessions loaded in {time.time()-t0:.1f}s")

        _clip_tokenize = clip.tokenize
        _clip_preprocess = _get_clip_preprocess()
        _inference_engine = "onnx"
        logger.info("✅ Inference engine: ONNX Runtime (optimized)")

    else:
        logger.info("⚠️ ONNX models not found, falling back to PyTorch...")
        import torch
        import clip

        t0 = time.time()
        try:
            model, preprocess = clip.load("ViT-B/32", device="cpu",
                                           download_root=CLIP_MODEL_DIR)
        except Exception:
            logger.warning("Could not load from cache, downloading...")
            model, preprocess = clip.load("ViT-B/32", device="cpu")

        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenize = clip.tokenize
        _inference_engine = "pytorch"
        logger.info(f"✅ Inference engine: PyTorch (loaded in {time.time()-t0:.1f}s)")


def get_text_embedding(text: str):
    """Encode text to 512-dim vector."""
    _init_inference()
    tokens = _clip_tokenize([text])

    if _inference_engine == "onnx":
        result = _onnx_text_session.run(None, {"input_ids": tokens.numpy()})[0]
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        return result[0]
    else:
        import torch
        with torch.no_grad():
            features = _clip_model.encode_text(tokens.to("cpu"))
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]


def get_image_embedding(image_file):
    """Encode image to 512-dim vector."""
    from PIL import Image
    _init_inference()
    image = _clip_preprocess(Image.open(image_file)).unsqueeze(0)

    if _inference_engine == "onnx":
        result = _onnx_image_session.run(None, {"pixel_values": image.numpy()})[0]
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        return result[0]
    else:
        import torch
        with torch.no_grad():
            features = _clip_model.encode_image(image.to("cpu"))
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]


# ==================================================
# Startup Event
# ==================================================

@app.on_event("startup")
async def startup_event():
    """Load models and build FAISS index at startup."""
    logger.info("🔄 Starting AI Search Engine (HF Spaces Edition)...")
    t0 = time.time()

    # 1. Initialize CLIP inference
    _init_inference()

    # 2. Build FAISS index from embeddings
    _build_faiss_index()

    elapsed = time.time() - t0
    logger.info(f"🎉 Server ready in {elapsed:.1f}s! ({len(_image_ids)} images indexed)")


# ==================================================
# API Endpoints
# ==================================================

# --- Settings (backward compat) ---
@app.post("/set-settings")
async def set_settings(settings: dict):
    return {"message": "Settings ignored in HF mode (env vars used)."}


# --- Image Router ---
@image_router.post("/search", response_model=SearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    logger.info(f"Image search request: {file.filename}")
    t0 = time.time()
    try:
        vector = get_image_embedding(file.file)
        t_embed = time.time()
        image_ids = search_faiss(vector, k=num_images)
        t_search = time.time()
        logger.info(f"  Timing: embed={t_embed-t0:.3f}s, search={t_search-t_embed:.3f}s, "
                     f"total={t_search-t0:.3f}s [{_inference_engine}]")
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Text Router ---
@text_router.post("/search", response_model=SearchResult)
async def search_text(request: TextSearchRequest):
    logger.info(f"Text search request: '{request.query}'")
    t0 = time.time()
    try:
        vector = get_text_embedding(request.query)
        t_embed = time.time()
        image_ids = search_faiss(vector, k=request.num_images)
        t_search = time.time()
        logger.info(f"  Timing: embed={t_embed-t0:.3f}s, search={t_search-t_embed:.3f}s, "
                     f"total={t_search-t0:.3f}s [{_inference_engine}]")
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Admin Router ---
@admin_router.get("/status")
async def admin_status():
    """Return current system status."""
    return {
        "inference_engine": _inference_engine or "not_loaded",
        "faiss_index_size": _faiss_index.ntotal if _faiss_index else 0,
        "total_images": len(_image_ids),
        "onnx_available": os.path.exists(
            os.path.join(ONNX_MODEL_DIR, "clip_text_encoder.onnx")
        ),
        "deployment": "huggingface-spaces"
    }


# --- Register Routers ---
app.include_router(image_router)
app.include_router(text_router)
app.include_router(admin_router)

# --- Static Files & Root ---
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def root():
    """Serve the frontend HTML."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": f"Welcome to the AI Search API (Frontend not found at {FRONTEND_DIR})"}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """Health check — used by UptimeRobot (HEAD) to keep the Space awake."""
    return {
        "status": "ok",
        "engine": _inference_engine or "not_loaded",
        "indexed": _faiss_index.ntotal if _faiss_index else 0,
        "deployment": "huggingface-spaces"
    }

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
