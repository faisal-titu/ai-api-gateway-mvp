# Personalized AI Image Search Engine 🔍

A powerful, multimodal search engine designed to explore **YOUR personal image collection**. By combining **OpenAI CLIP** (semantic understanding) and **AWS OpenSearch**, it enables you to search your own photos using natural language ("a birthday party", "my cat sleeping") or finding similar visual styles.

🔴 **Live Demo:** [http://13.63.120.96:8000/](http://13.63.120.96:8000/)

This project features a modern **FastAPI** backend and a **Tailwind CSS** frontend with Glassmorphism UI.

---

## 🚀 Features

-   **Semantic Text Search**: "Find a sad dog in the rain" (Understands context, not just keywords).
-   **Reverse Image Search**: Upload an image to find visually similar results.
-   **High Performance**: Uses approximate nearest neighbor search (k-NN) for sub-second retrieval from large datasets.
-   **Modern Architecture**:
    -   **Frontend**: HTML5, Tailwind CSS, Vanilla JS (No build step required).
    -   **Backend**: Python FastAPI with async processing.
    -   **AI**: OpenAI CLIP (ViT-B/32) running on CPU (optimized with Torch).
    -   **Database**: OpenSearch with k-NN plugin.

---

## 🛠️ Prerequisites

-   **Docker Desktop** (or Docker Engine + Compose)
-   **Git**
-   **Python 3.9+** (Optional, for local scripting)
-   **AWS CLI** (Optional, for deployment or S3 access)

---

## 💻 Local Setup & Execution Guide

Follow these steps to run the entire stack locally.

### 1. Clone the Repository
```bash
git clone https://github.com/faisal-titu/ai-api-gateway-mvp.git
cd ai-api-gateway-mvp
```

### 2. Configure Environment Variables
Create a `.env.aws` file (used by Docker Compose) with the following content:

```bash
# .env.aws
OPENSEARCH_HOST=opensearch-node1
OPENSEARCH_PORT=9200
```
*Note: `opensearch-node1` is the hostname within the Docker network.*

### 3. Start the Application (Docker Compose)
Run the following command to build the API container and start OpenSearch:

```bash
docker-compose -f docker-compose.aws.yml up --build
```
-   **Wait 30-60 seconds** for OpenSearch to fully initialize.
-   The API will be available at imports `http://localhost:8000`.

### 4. Verify System Health
Open a new terminal and run:
```bash
curl http://localhost:8000/health
```
**Expected Output:**
```json
{"status": "ok"}
```

---

## 🧠 Indexing Data (The "Embeddings")

To search images, the system needs to know their vector representations.
**Prerequisite**: You must have a file `datalake/embeddings.jsonl` containing pre-computed CLIP embeddings. If you don't have this, use the scripts in `dev/` to generate it.

**Example data format (JSONL):**
```json
{"image_id": "image123", "embedding": [0.123, -0.456, ...]}
```

### Run Bulk Indexing
Use the API to load these embeddings into OpenSearch:

```bash
curl -X POST "http://localhost:8000/images/bulk-index-embeddings?index_name=unsplash_images&file_path=/app/datalake/embeddings.jsonl"
```
*This endpoint efficiently indexes thousands of vectors in seconds.*

---

## 🌐 Frontend Access

Once the containers are running, open your browser:
👉 **http://localhost:8000/**

You will see the **AI Search Landing Page**.
-   **Authentication**: None required for MVP.
-   **Images**: Displayed directly from S3 (configured in `frontend/script.js`).

---

## 📚 API Reference

Here is the detailed documentation for all available endpoints.

### 1. Health Check
**GET /health**
-   **Description**: Checks if the API is running.
-   **Response**: `{"status": "ok"}`

### 2. Text Search
**POST /texts/search**
-   **Description**: Semantically searching images using a text query.
-   **Body**:
    ```json
    {
      "query": "sunset over mountains",
      "num_images": 5
    }
    ```
-   **Curl Example**:
    ```bash
    curl -X POST "http://localhost:8000/texts/search" \
         -H "Content-Type: application/json" \
         -d '{"query": "cyberpunk city", "num_images": 10}'
    ```
-   **Response**:
    ```json
    {
      "image_ids": ["id_1", "id_2", "id_3"]
    }
    ```

### 3. Image Search (Reverse Search)
**POST /images/search**
-   **Description**: Finds similar images to an uploaded file.
-   **Form Data**:
    -   `file`: The image file to search with.
    -   `num_images`: (Optional) Number of results (default 5).
-   **Curl Example**:
    ```bash
    curl -X POST "http://localhost:8000/images/search" \
         -F "file=@/path/to/my_image.jpg" \
         -F "num_images=5"
    ```
-   **Response**: Same as Text Search (`image_ids` list).

### 4. Bulk Indexing
**POST /images/bulk-index-embeddings**
-   **Description**: Indexes a large JSONL file of embeddings into OpenSearch.
-   **Query Params**:
    -   `index_name`: Name of the OpenSearch index (e.g., `unsplash_images`).
    -   `file_path`: Internal path to the file (must be inside Docker container, e.g., `/app/datalake/embeddings.jsonl`).
-   **Curl Example**: See "Indexing Data" section above.

### 5. Settings (Placeholder)
**POST /set-settings**
-   **Description**: Placeholder endpoint for configuration (currently uses ENV vars).

---

## 📂 Project Structure

```
ai-api-gateway-mvp/
├── app/
│   ├── api/
│   │   └── fastapi_aws.py       # Main Application Logic
├── frontend/                    # Source code for the UI
│   ├── index.html               # Main Landing Page
│   ├── script.js                # Frontend Logic (API calls)
│   └── style.css                # Custom Styles (Tailwind overrides)
├── datalake/                    # Volume-mounted data folder (embeddings.jsonl)
├── dev/                         # Development scripts & legacy features
├── docker-compose.aws.yml       # Docker services configuration
├── Dockerfile.aws               # Build instructions for API container
└── requirements.txt             # Python dependencies
```

---

## ❓ Troubleshooting

### 1. OpenSearch Connection Error
-   **Symptom**: `ConnectionRefusedError` or timeout in logs.
-   **Fix**: OpenSearch takes time to start. Wait 60 seconds after `docker-compose up`. Ensure `OPENSEARCH_HOST` in `.env.aws` matches the service name in `docker-compose.aws.yml`.

### 2. Frontend not updating
-   **Symptom**: Old styles visible.
-   **Fix**: Perform a **Hard Refresh** (Ctrl+Shift+R) in your browser. Since we use a Docker volume mount, changes to `frontend/` files are instant.

### 3. "CircuitBreakerException" in OpenSearch
-   **Symptom**: Bulk indexing fails with memory error.
-   **Fix**: We have configured `OPENSEARCH_JAVA_OPTS="-Xms512m -Xmx512m"` in `docker-compose.aws.yml` and reduced indexing chunk size to 50. Ensure your Docker machine has at least 2GB RAM allocated.

---

## 📜 License
MIT License.
