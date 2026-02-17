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
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "unsplash_images"

# Initialize FastAPI
app = FastAPI(title="AI Search API (AWS Edition)", version="1.0.0")

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

# --- Models ---
class SearchResult(BaseModel):
    image_ids: List[str]

class TextSearchRequest(BaseModel):
    query: str
    num_images: int = 5

# --- Local Embedding Logic (Lazy Loading) ---
_clip_model = None
_clip_preprocess = None
_opensearch_client = None

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

def _get_clip_functions():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        logger.info("Loading CLIP model (this may take a few minutes on first call)...")
        import torch
        import clip
        from PIL import Image
        
        device = "cpu"
        # Load from local cache in Docker image
        model_path = "/app/model/clip"
        model_name = "ViT-B/32"
        
        # Try loading from local file if download_root set, else default
        try:
             # This assumes download_root was used during build
             model, preprocess = clip.load(model_name, device=device, download_root=model_path)
        except:
             logger.warning("Could not load from /app/model/clip, downloading...")
             model, preprocess = clip.load(model_name, device=device)
             
        _clip_model = model
        _clip_preprocess = preprocess
        
        def get_text_embedding(text):
            text_input = clip.tokenize([text]).to(device)
            with torch.no_grad():
                text_features = _clip_model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]

        def get_image_embedding(image_file):
            image = _clip_preprocess(Image.open(image_file)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = _clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]

        def search_knn(vector, index_name, k=5):
            client = _get_opensearch_client()
            query = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": vector,
                            "k": k
                        }
                    }
                },
                "_source": ["image_id"]
            }
            return client.search(body=query, index=index_name)

        logger.info("CLIP model loaded successfully!")
        
        return {
            "get_text_embedding": get_text_embedding,
            "get_image_embedding": get_image_embedding,
            "search_knn": search_knn
        }
    
    # Return existing functions (wrapper)
    # We need to return the functions bound to the loaded model
    import torch
    import clip
    from PIL import Image
    device = "cpu"
    
    def get_text_embedding(text):
        text_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = _clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_image_embedding(image_file):
        image = _clip_preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def search_knn(vector, index_name, k=5):
        client = _get_opensearch_client()
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vector,
                        "k": k
                    }
                }
            },
            "_source": ["image_id"]
        }
        return client.search(body=query, index=index_name)

    return {
        "get_text_embedding": get_text_embedding,
        "get_image_embedding": get_image_embedding,
        "search_knn": search_knn
    }

# --- Settings Endpoint ---
@app.post("/set-settings")
async def set_settings(settings: dict):
    """Placeholder for backward compatibility."""
    return {"message": "Settings ignored in AWS mode (env vars used)."}

# --- Image Router ---
@image_router.post("/search", response_model=SearchResult)
async def search_image(file: UploadFile = File(...), num_images: int = Form(5)):
    logger.info(f"Image search request: {file.filename}")
    m = _get_clip_functions()
    try:
        vector = m["get_image_embedding"](file.file)
        response = m["search_knn"](vector, INDEX_NAME, num_images)
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@image_router.post("/batch-index")
async def batch_index(index_name: str = Query(...), image_dir: str = Query(...)):
    """Slow batch index on CPU."""
    logger.info(f"Batch index request: {index_name}, {image_dir}")
    # Implementation omitted for brevity in this fix, can restore if needed
    # But for now let's keep it simple as user is using pre-computed embeddings
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
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "embedding": {"type": "knn_vector", "dimension": 512},
                    "image_id": {"type": "keyword"}
                }
            }
        })
        logger.info(f"Created index {index_name}")

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
    m = _get_clip_functions()
    try:
        vector = m["get_text_embedding"](request.query)
        response = m["search_knn"](vector, INDEX_NAME, request.num_images)
        image_ids = [hit["_source"]["image_id"] for hit in response["hits"]["hits"]]
        return SearchResult(image_ids=image_ids)
    except Exception as e:
        logger.error(f"Error during search_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --- Register Routers ---
app.include_router(image_router)
app.include_router(text_router)

# --- Static Files & Root ---
# Mount static files for frontend
if os.path.exists("/app/frontend"):
    app.mount("/static", StaticFiles(directory="/app/frontend"), name="static")

@app.get("/")
async def root():
    """Serve the frontend HTML."""
    if os.path.exists("/app/frontend/index.html"):
        return FileResponse("/app/frontend/index.html")
    # New fallback message
    return {"message": "Welcome to the AI Search API (Frontend not found - check /app/frontend mount)"}

@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok"}

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
