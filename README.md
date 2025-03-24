# FastAPI AI Search

This project implements a FastAPI application for multimodal search, allowing users to perform text-to-image and image-to-image searches.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
  - [Using Python](#using-python)
  - [Using Docker](#using-docker)
- [Endpoints](#endpoints)
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
- [License](#license)
- [Instructions to Update](#instructions-to-update)

## Prerequisites
<a id="prerequisites"></a>

- Python 3.8 or higher
- Pip (Python package installer)
- Docker (if you choose to run the application using Docker)

## Installation
<a id="installation"></a>

1. **Clone the repository:**

   Open your terminal and run:

   ```bash
   git clone git@bitbucket.org:resilientsage/ai-api-gateway-mvp.git
   ```

2. **Go to the directory:**

   ```bash
   cd ai-api-gateway-mvp
   ```

3. **Install the requirements:**

   Ensure you have a virtual environment activated (optional but recommended), then install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
<a id="running-the-application"></a>
You can run the FastAPI application in two ways: directly with Python or using Docker.

### Using Python
<a id="using-python"></a>
Run the FastAPI application using the following command:

```bash
uvicorn app.api.fastapi_ai_search:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
<a id="using-docker"></a>
To run the FastAPI application using Docker, follow these steps:

1. **Build the Docker image:**

   From the root directory of the project, run:

   ```bash
   docker build -t fastapi-ai-search .
   ```

2. **Run the Docker container:**

   ```bash
   docker run --name fastapi-ai-search --network without_security_opensearch-net -p 8000:8000 fastapi-ai-search
   ```

## Endpoints
<a id="endpoints"></a>

### Text to Image Search
<a id="text-to-image-search"></a>
- **Description:** Perform a text-to-image search.
- **Endpoint:** `/texts/search`
- **Method:** POST
- **Input:**
  - `query` (string): The text query.
  - `num_images` (int): Number of images to return.
- **Output:** JSON object containing search results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/texts/search" -H "accept: application/json" -H "Content-Type: application/json" -d '{"query": "A beautiful sunset over the mountains", "num_images": 5}'
```

### Batch Index Texts
<a id="batch-index-texts"></a>
- **Description:** Index a batch of text data.
- **Endpoint:** `/texts/batch-index`
- **Method:** POST
- **Input:**
  - `texts` (list of strings): The texts to index.
- **Output:** JSON object containing indexing results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/texts/batch-index" -H "accept: application/json" -H "Content-Type: application/json" -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

### Image to Image Search
<a id="image-to-image-search"></a>
- **Description:** Perform an image-to-image search.
- **Endpoint:** `/images/search`
- **Method:** POST
- **Input:**
  - `file` (file): The image file to search.
  - `num_images` (int): Number of images to return.
- **Output:** JSON object containing search results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/images/search" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg" -F "num_images=5"
```

### Batch Index Images
<a id="batch-index-images"></a>
- **Description:** Index a batch of image data.
- **Endpoint:** `/images/batch-index`
- **Method:** POST
- **Input:**
  - `image_dir` (string): Directory containing images.
- **Output:** JSON object containing indexing results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/images/batch-index" -H "accept: application/json" -d '{"image_dir": "/path/to/images"}'
```

### Face Detection in Single Image
<a id="face-detection-in-single-image"></a>
- **Description:** Detect faces in a single image.
- **Endpoint:** `/faces/detection/single`
- **Method:** POST
- **Input:**
  - `file` (file): The image file to detect faces in.
- **Output:** JSON object containing detected faces and their bounding boxes.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/faces/detection/single" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

### Face Detection in Batch
<a id="face-detection-in-batch"></a>
- **Description:** Detect faces in a batch of images.
- **Endpoint:** `/faces/detection/batch`
- **Method:** POST
- **Input:**
  - `image_dir` (string): Directory containing images.
  - `face_crop_dir` (string): Directory to save cropped faces.
- **Output:** JSON object containing details of detected faces for each image.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/faces/detection/batch" -H "accept: application/json" -d '{"image_dir": "/path/to/images", "face_crop_dir": "/path/to/save/cropped/faces"}'
```

### Face Embedding for Single Image
<a id="face-embedding-for-single-image"></a>
- **Description:** Generate face embeddings for a single image.
- **Endpoint:** `/faces/embedding/single`
- **Method:** POST
- **Input:**
  - `file` (file): The image file to generate embeddings for.
  - `index_name` (string, optional): The index name to store embeddings.
- **Output:** JSON object containing face embeddings.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/faces/embedding/single" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg" -F "index_name=your_index_name"
```

### Face Embedding for Batch
<a id="face-embedding-for-batch"></a>
- **Description:** Generate face embeddings for a batch of images.
- **Endpoint:** `/faces/embedding/batch`
- **Method:** POST
- **Input:**
  - `image_dir` (string): Directory containing images.
  - `face_crop_dir` (string): Directory to save cropped faces.
  - `index_name` (string): The index name to store embeddings.
  - `batch_size` (int): Batch size for processing.
- **Output:** JSON object containing details of generated embeddings.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/faces/embedding/batch" -H "accept: application/json" -d '{"image_dir": "/path/to/images", "face_crop_dir": "/path/to/save/cropped/faces", "index_name": "your_index_name", "batch_size": 32}'
```

### Face Search
<a id="face-search"></a>
- **Description:** Search for similar faces in an image.
- **Endpoint:** `/faces/search`
- **Method:** POST
- **Input:**
  - `file` (file): The image file to search for similar faces.
  - `index_name` (string): The index name to search in.
  - `k` (int): Number of similar faces to return.
- **Output:** JSON object containing search results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/faces/search" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg" -F "index_name=your_index_name" -F "k=5"
```

### Combined Search
<a id="combined-search"></a>
- **Description:** Perform a combined search using multiple modalities.
- **Endpoint:** `/combined/search`
- **Method:** POST
- **Input:**
  - `query` (string): The text query.
  - `file` (file): The image file to search.
  - `num_results` (int): Number of results to return.
- **Output:** JSON object containing search results.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/combined/search" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "query=A beautiful sunset over the mountains" -F "file=@/path/to/your/image.jpg" -F "num_results=5"
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Instructions to Update
Replace username and existing-repo in the clone command with your actual GitHub or Bitbucket username and repository name.