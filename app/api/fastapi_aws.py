import os
import time
import logging
from typing import List, Optional

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
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "unsplash_images"
ONNX_MODEL_DIR = os.getenv("ONNX_MODEL_DIR", os.path.join(APP_BASE_DIR, "model", "clip_onnx"))
CLIP_MODEL_DIR = os.getenv("CLIP_MODEL_DIR", os.path.join(APP_BASE_DIR, "model", "clip"))
FRONTEND_DIR = os.getenv("FRONTEND_DIR", os.path.join(APP_BASE_DIR, "frontend"))

# Initialize FastAPI
app = FastAPI(title="AI Search API (AWS Edition)", version="2.0.0")

# CORS (Allow frontend access)
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

# --- Lazy-loaded Globals ---
_opensearch_client = None
_inference_engine = None  # Will be "onnx" or "pytorch"
_onnx_text_session = None
_onnx_image_session = None
_clip_model = None
_clip_preprocess = None
_clip_tokenize = None


def _get_opensearch_client():
    global _opensearch_client
    if _opensearch_client is None:
        logger.info("Initializing OpenSearch client...")
        from opensearchpy import OpenSearch
        _opensearch_client = OpenSearch(
            hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=30
        )
        logger.info("OpenSearch client ready.")
    return _opensearch_client


def _init_inference():
    """Initialize the inference engine: ONNX if available, else PyTorch."""
    global _inference_engine, _onnx_text_session, _onnx_image_session
    global _clip_model, _clip_preprocess, _clip_tokenize

    if _inference_engine is not None:
        return  # Already initialized

    import clip
    from PIL import Image

    text_onnx = os.path.join(ONNX_MODEL_DIR, "clip_text_encoder.onnx")
    image_onnx = os.path.join(ONNX_MODEL_DIR, "clip_image_encoder.onnx")

    if os.path.exists(text_onnx) and os.path.exists(image_onnx):
        # --- ONNX Runtime Path (FAST) ---
        logger.info("🚀 ONNX models found! Using ONNX Runtime for inference.")
        import onnxruntime as ort

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

        # Still need CLIP for tokenizer and image preprocessing
        _clip_tokenize = clip.tokenize

        # Load just the preprocessing transform (lightweight)
        _, _clip_preprocess = clip.load("ViT-B/32", device="cpu",
                                         download_root=CLIP_MODEL_DIR)
        _inference_engine = "onnx"
        logger.info("✅ Inference engine: ONNX Runtime (optimized)")

    else:
        # --- PyTorch Fallback (SLOW) ---
        logger.info("⚠️ ONNX models not found, falling back to PyTorch...")
        import torch

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
    """Encode text to 512-dim vector using ONNX or PyTorch."""
    import numpy as np
    _init_inference()

    tokens = _clip_tokenize([text])

    if _inference_engine == "onnx":
        result = _onnx_text_session.run(
            None, {"input_ids": tokens.numpy()}
        )[0]
        # Normalize
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        return result[0]
    else:
        import torch
        with torch.no_grad():
            features = _clip_model.encode_text(tokens.to("cpu"))
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]


def get_image_embedding(image_file):
    """Encode image to 512-dim vector using ONNX or PyTorch."""
    import numpy as np
    from PIL import Image
    _init_inference()

    image = _clip_preprocess(Image.open(image_file)).unsqueeze(0)

    if _inference_engine == "onnx":
        result = _onnx_image_session.run(
            None, {"pixel_values": image.numpy()}
        )[0]
        # Normalize
        result = result / np.linalg.norm(result, axis=-1, keepdims=True)
        return result[0]
    else:
        import torch
        with torch.no_grad():
            features = _clip_model.encode_image(image.to("cpu"))
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]


def search_knn(vector, index_name, k=5):
    """Run k-NN search on OpenSearch."""
    client = _get_opensearch_client()
    query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector.tolist() if hasattr(vector, 'tolist') else list(vector),
                    "k": k
                }
            }
        },
        "_source": ["image_id"]
    }
    return client.search(body=query, index=index_name)


# --- Settings Endpoint ---
@app.post("/set-settings")
async def set_settings(settings: dict):
    """Placeholder for backward compatibility."""
    return {"message": "Settings ignored in AWS mode (env vars used)."}


# --- Image Router ---
@image_router.post("/search", response_model=SearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    logger.info(f"Image search request: {file.filename}")
    t0 = time.time()
    try:
        vector = get_image_embedding(file.file)
        t_embed = time.time()
        response = search_knn(vector, INDEX_NAME, num_images)
        t_search = time.time()
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        logger.info(f"  Timing: embed={t_embed-t0:.3f}s, search={t_search-t_embed:.3f}s, "
                     f"total={t_search-t0:.3f}s [{_inference_engine}]")
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@image_router.post("/batch-index")
async def batch_index(index_name: str = Query(...), image_dir: str = Query(...)):
    """Slow batch index on CPU."""
    logger.info(f"Batch index request: {index_name}, {image_dir}")
    return {"message": "Use /bulk-index-embeddings for faster indexing on CPU."}


@image_router.post("/bulk-index-embeddings")
async def bulk_index_from_file(index_name: str = Query(...), file_path: str = Query(...)):
    """Bulk index pre-computed embeddings from a JSONL file."""
    logger.info(f"Bulk index from file: index={index_name}, file={file_path}")
    client = _get_opensearch_client()

    import json
    from opensearchpy.helpers import bulk

    # Check if index exists
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body={
            "settings": {
                "index.knn": True,
                "index.knn.algo_param.ef_search": 256 
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 512,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 16
                            }
                        }
                    },
                    "image_id": {"type": "keyword"}
                }
            }
        })
        logger.info(f"Created index {index_name} with optimized HNSW settings")

    def generate_actions():
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield {
                    "_index": index_name,
                    "_source": {
                        "image_id": data["image_id"],
                        "embedding": data["embedding"]
                    }
                }

    try:
        success, failed = bulk(client, generate_actions(), stats_only=True, chunk_size=50)
        logger.info(f"File indexing complete: {success} indexed, {failed} errors")
        return {"indexed": success, "errors": failed}
    except Exception as e:
        logger.error(f"Error during bulk index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Text Router ---
@text_router.post("/search", response_model=SearchResult)
async def search_text(request: TextSearchRequest):
    logger.info(f"Text search request: '{request.query}'")
    t0 = time.time()
    try:
        vector = get_text_embedding(request.query)
        t_embed = time.time()
        response = search_knn(vector, INDEX_NAME, request.num_images)
        t_search = time.time()
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        logger.info(f"  Timing: embed={t_embed-t0:.3f}s, search={t_search-t_embed:.3f}s, "
                     f"total={t_search-t0:.3f}s [{_inference_engine}]")
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Admin Router ---
@admin_router.post("/optimize-index")
async def optimize_index(index_name: str = Query(default="unsplash_images")):
    """Force merge index and tune HNSW parameters for maximum speed."""
    logger.info(f"Optimizing index: {index_name}")
    client = _get_opensearch_client()

    results = {}

    # 1. Force merge to 1 segment (eliminates segment overhead)
    try:
        client.indices.forcemerge(index=index_name, max_num_segments=1)
        results["force_merge"] = "success"
        logger.info("  Force merge complete")
    except Exception as e:
        results["force_merge"] = f"error: {e}"
        logger.error(f"  Force merge failed: {e}")

    # 2. Refresh index
    try:
        client.indices.refresh(index=index_name)
        results["refresh"] = "success"
    except Exception as e:
        results["refresh"] = f"error: {e}"

    # 3. Warm up k-NN by loading graphs into memory
    try:
        client.transport.perform_request(
            "GET",
            f"/_plugins/_knn/warmup/{index_name}"
        )
        results["knn_warmup"] = "success"
        logger.info("  k-NN warmup complete")
    except Exception as e:
        results["knn_warmup"] = f"error: {e}"
        logger.warning(f"  k-NN warmup failed (may not be supported): {e}")

    # 4. Get index stats
    try:
        stats = client.indices.stats(index=index_name)
        doc_count = stats["indices"][index_name]["primaries"]["docs"]["count"]
        size_bytes = stats["indices"][index_name]["primaries"]["store"]["size_in_bytes"]
        results["docs"] = doc_count
        results["size_mb"] = round(size_bytes / 1024 / 1024, 1)
    except Exception as e:
        results["stats"] = f"error: {e}"

    results["engine"] = _inference_engine or "not_loaded"
    logger.info(f"  Optimization complete: {results}")
    return results


@admin_router.get("/status")
async def admin_status():
    """Return current system status including inference engine."""
    return {
        "inference_engine": _inference_engine or "not_loaded",
        "opensearch_host": OPENSEARCH_HOST,
        "index_name": INDEX_NAME,
        "onnx_available": os.path.exists(
            os.path.join(ONNX_MODEL_DIR, "clip_text_encoder.onnx")
        )
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

@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok", "engine": _inference_engine or "not_loaded"}

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
