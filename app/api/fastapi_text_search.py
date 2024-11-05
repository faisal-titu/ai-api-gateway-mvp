from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from typing import List, Union
import glob

# Import necessary functions and clients
from dev.embedding_generation import get_text_embedding, get_image_embedding
from dev.search import search_knn
from dev.open_search_client import client
import tempfile
from PIL import Image
import PIL
import io

# Initialize FastAPI app
app = FastAPI()

# Index and image directory configuration
INDEX_NAME = 'index_02'
IMAGE_DIR = "/home/iot/Desktop/search/dataset2/unsplash"

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
    

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Image Search API. Use POST /search-text to search images."}

@app.post("/search-text", response_model=SearchResult)
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

@app.post("/search-image", response_model=ImageSearchResult)
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



class EmbeddingResponse(BaseModel):
    image_id: str
    embedding: List[float]

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingResponse]

class SingleEmbeddingResponse(BaseModel):
    embedding: List[float]

@app.post("/embed-image-batch", response_model=BatchEmbeddingResponse)
async def embed_image_batch(files: List[UploadFile] = File(...)):
    """
    Accepts a batch of up to 5 images and returns their embeddings.
    """
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Only 5 images allowed per batch.")

    embeddings = []
    
    for file in files:
        try:
            # Process each image
            file_content = await file.read()
            image = Image.open(io.BytesIO(file_content)).convert("RGB")

            # Save and process the image
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                image.save(tmp, format="JPEG")
                tmp.seek(0)
                image_vector = get_image_embedding(tmp.name)

            embeddings.append(EmbeddingResponse(image_id=file.filename, embedding=image_vector))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {file.filename}: {e}")

    return BatchEmbeddingResponse(embeddings=embeddings)




class TextEmbeddingResponse(BaseModel):
    text_part: str
    embedding: List[float]

class BatchTextEmbeddingResponse(BaseModel):
    embeddings: List[TextEmbeddingResponse]

@app.post("/embed-text-batch", response_model=BatchTextEmbeddingResponse)
async def embed_text_batch(texts: List[str] = Form(...)):
    """
    Accepts a batch of up to 5 text parts and returns their embeddings.
    """
    if len(texts) > 5:
        raise HTTPException(status_code=400, detail="Only 5 text parts allowed per batch.")

    embeddings = []
    
    for text in texts:
        try:
            embedding = get_text_embedding(text)
            embeddings.append(TextEmbeddingResponse(text_part=text, embedding=embedding))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing text part: {e}")

    return BatchTextEmbeddingResponse(embeddings=embeddings)


# Run this FastAPI app using Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
