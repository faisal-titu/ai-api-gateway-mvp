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
   - [Text to Image Search](#text-to-image-search)  
   - [Batch Index Texts](#batch-index-texts)  
   - [Image to Image Search](#image-to-image-search)  
   - [Batch Index Images](#batch-index-images)  
   - [Face Detection in Single Image](#face-detection-in-single-image)  
   - [Face Detection in Batch](#face-detection-in-batch)  
   - [Face Embedding for Single Image](#face-embedding-for-single-image)  
   - [Face Embedding for Batch](#face-embedding-for-batch)  
   - [Face Search](#face-search)  
   - [Combined Search](#combined-search)  
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

### Text to Image Search
- **Description:** Perform a text-to-image search.  
- **Endpoint:** `/texts/search`  
- **Method:** POST  

**Example Request:**  
```bash
curl -X POST "http://127.0.0.1:8000/texts/search" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"query": "A beautiful sunset over the mountains", "num_images": 5}'
```

---

### Batch Index Texts
- **Description:** Index a batch of text data.  
- **Endpoint:** `/texts/batch-index`  
- **Method:** POST  

**Example Request:**  
```bash
curl -X POST "http://127.0.0.1:8000/texts/batch-index" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

---

### Image to Image Search
- **Description:** Perform an image-to-image search.  
- **Endpoint:** `/images/search`  
- **Method:** POST  

**Example Request:**  
```bash
curl -X POST "http://127.0.0.1:8000/images/search" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/image.jpg" \
-F "num_images=5"
```

---

### [Other Endpoints](#endpoints)  

See the full list of endpoints for additional functionality such as face detection, embeddings, and combined searches.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Instructions to Update
Replace placeholders in the clone command with your Bitbucket username and repository details.

--- 

This format uses markdown properly, ensuring a clean and professional layout on Bitbucket.