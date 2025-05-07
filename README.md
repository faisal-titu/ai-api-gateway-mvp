Here's a properly formatted version of your README for Bitbucket:

---

# FastAPI AI Search

This project implements a FastAPI application for multimodal search, allowing users to perform text-to-image and image-to-image searches.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Running the Application](#running-the-application)  
   - [Using Python](#using-python)  
   - [Using Docker](#using-docker)  
4. [Endpoints](#endpoints)  
5. [License](#license)  
6. [Instructions to Update](#instructions-to-update)  

---

## Prerequisites
- Python 3.8 or higher  
- Pip (Python package installer)  
- Docker (if you choose to run the application using Docker)  

---

## Installation

1. **Clone the repository:**  
   ```bash
   git clone git@bitbucket.org:resilientsage/ai-api-gateway-mvp.git
   ```
   
2. **Go to the project directory:**  
   ```bash
   cd ai-api-gateway-mvp
   ```
   
3. **Install the requirements:**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Using Python
Run the FastAPI application directly with Python:  
```bash
uvicorn app.api.fastapi_ai_search:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
Run the FastAPI application using Docker:  

1. **Build the Docker image:**  
   ```bash
   docker build -t fastapi-ai-search .
   ```

2. **Run the Docker container:**  
   ```bash
   docker run --name fastapi-ai-search --network without_security_opensearch-net -p 8000:8000 fastapi-ai-search
   ```

---

## Endpoints

Below is a detailed reference for every FastAPI endpoint. Each entry includes:
- **What** the endpoint does
- **How** to call it (HTTP method, URL, headers)
- **Example Request** (cURL)
- **Example Response**

---

### Root
**GET /**

What: Health check and welcome message.

How:
```bash
curl -X GET http://localhost:8000/
```
Example Response:
```json
{ "message": "Welcome to the AI Search API" }
```

---

### Settings
**POST /set-settings**

What: Configure which OpenSearch index and image directory to use for subsequent searches and batch operations.

How:
```bash
curl -X POST http://localhost:8000/set-settings \
  -H "Content-Type: application/json" \
  -d '{
        "index_name": "my_index",
        "image_dir": "/app/datalake/images"
      }'
```
Example Response:
```json
{
  "message": "Settings updated successfully",
  "index_name": "my_index",
  "image_dir": "/app/datalake/images"
}
```

---

### Metrics
**GET /metrics**

What: Exposes Prometheus metrics for monitoring (latency, request counts, custom histograms).

How:
```bash
curl -X GET http://localhost:8000/metrics
```

**GET /test-metrics**

What: Simple counter endpoint for testing instrumentation.

How:
```bash
curl -X GET http://localhost:8000/test-metrics
```
Example Response:
```
# HELP test_counter_total Test counter for debugging
# TYPE test_counter_total counter
... 1.0
```

---

### Text Search
**POST /texts/search**

What: Proxy text query to TorchServe CLIP for text-to-image search.

How:
```bash
curl -X POST http://localhost:8000/texts/search \
  -H "Content-Type: application/json" \
  -d '{
        "query": "a photo of a cat",
        "num_images": 5
      }'
```
Example Response:
```json
{ "image_ids": ["gQFZxLe3m4g", "pesu5W2yXmQ", "5py1uD3sxNA", ...] }
```

---

### Batch Index Text
**POST /texts/batch-index**

What: Read a JSON-lines file of text metadata, generate embeddings locally, and bulk-index into OpenSearch.

How:
```bash
curl -X POST "http://localhost:8000/texts/batch-index?index_name=my_text_index&text_file_path=/app/datalake/meta.jsonl"
```

---

### Image Search
**POST /images/search**

What: Proxy image file to TorchServe CLIP for image-to-image search.

How:
```bash
curl -X POST http://localhost:8000/images/search \
  -F "file=@/path/to/image.jpg" \
  -F "num_images=5"
```
Example Response:
```json
{
  "query_image_id": "image.jpg",
  "similar_image_ids": ["GH_b1WXHKbU", "hfs2ierY1mY", ...]
}
```

---

### Batch Index Images
**POST /images/batch-index**

What: Walk a folder of images inside the container, generate CLIP embeddings locally, and bulk-index into OpenSearch.

How:
```bash
curl -X POST "http://localhost:8000/images/batch-index?index_name=my_image_index&image_dir=/app/datalake/small_1000"
```

---

### Face Detection (Single)
**POST /faces/detection/single**

What: Detect and crop faces in one uploaded image.

How:
```bash
curl -X POST http://localhost:8000/faces/detection/single \
  -F "file=@/path/to/portrait.jpg"
```
Example Response (one face):
```json
{
  "status": "Faces detected",
  "image_name": "portrait.jpg",
  "total_faces_detected": 1,
  "processed_images": [
    {
      "bounding_box": [x1, y1, x2, y2],
      "confidence": 0.99,
      "cropped_width": 150,
      "cropped_height": 150,
      "cropped_resolution": "150x150",
      "cropped_image_path": "datalake/cropped_faces/1234_face_1.jpg"
    }
  ]
}
```

---

### Face Detection (Batch)
**POST /faces/detection/batch**

What: Detect and crop faces in every image under a directory.

How:
```bash
curl -X POST "http://localhost:8000/faces/detection/batch?image_dir=/app/facenet_data/original_images&face_crop_dir=/app/facenet_data/crops"
```

---

### Face Embedding (Single)
**POST /faces/embedding/single**

What: Detect, embed, and optionally index faces in one image.

How:
```bash
curl -X POST "http://localhost:8000/faces/embedding/single?index_name=my_face_index" \
  -F "file=@/path/to/portrait.jpg"
```

---

### Face Embedding (Batch)
**POST /faces/embedding/batch**

What: Detect, embed, and bulk-index faces from all cropped images in a folder.

How:
```bash
curl -X POST "http://localhost:8000/faces/embedding/batch?image_dir=/app/facenet_data/original_images&face_crop_dir=/app/facenet_data/crops&index_name=my_face_index&batch_size=32"
```

---

### Face Search
**POST /faces/search**

What: Given an image, detect and embed faces locally, then perform KNN lookup in OpenSearch index.

How:
```bash
curl -X POST "http://localhost:8000/faces/search?index_name=my_face_index&k=5" \
  -F "file=@/path/to/portrait.jpg"
```
Example Response:
```json
{
  "status": "Search completed",
  "image_name": "portrait.jpg",
  "total_faces_detected": 1,
  "search_results": [
    {
      "face_id": "abcd1234",
      "box": [x1,y1,x2,y2],
      "similar_faces": [ {"image_id":"img1","face_id":"f1","score":0.95}, ... ]
    }
  ]
}
```

---

### Combined Search
**POST /combined/search**

What: In parallel, perform image-based search and face-based search for one upload.

How:
```bash
curl -X POST "http://localhost:8000/combined/search?image_index=my_image_index&face_index=my_face_index&num_images=5" \
  -F "file=@/path/to/portrait.jpg"
```
Example Response:
```json
{
  "status": "success",
  "details": {
    "image_search": { "query_image_id":"portrait.jpg","similar_image_ids":["id1",... ]},
    "face_search": { ... }
  }
}
```

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Instructions to Update
Replace placeholders in the clone command with your Bitbucket username and repository details.

--- 

This format uses markdown properly, ensuring a clean and professional layout on Bitbucket.