# FastAPI AI Search

This project implements a FastAPI application for multimodal search, allowing users to perform text-to-image and image-to-image searches.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using Docker](#using-docker)
- [Making Requests](#making-requests)
  - [Text to Image Search](#text-to-image-search)
  - [Image to Image Search](#image-to-image-search)

## Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- Docker (if you choose to run the application using Docker)

## Installation

1. **Clone the repository:**

   Open your terminal and run:

```bash
   git clone git@bitbucket.org:resilientsage/ai-api-gateway-mvp.git
```
Replace username with the GitHub account name and existing-repo with the name of the repository.

Go to the directory:
```bash
cd existing-repo
```
Install the requirements:

Ensure you have a virtual environment activated (optional but recommended), then install the necessary packages:
```bash
pip install -r requirements.txt
```

## Running the Application
You can run the FastAPI application in two ways: directly with Python or using Docker.

Using Python
Run the FastAPI application using the following command:
```bash
uvicorn fastapi_text_search:app --host 0.0.0.0 --port 8000 --reload
```
Using Docker
To run the FastAPI application using Docker, follow these steps:

Build the Docker image:

From the root directory of the project, run:
```bash
docker build -t fastapi-ai-search .
```

Run the Docker container:
```bash
docker run --name fastapi-ai-search --network without_security_opensearch-net -p 8000:8000 fastapi-ai-search
```
## Making Requests
Once the application is running, you can make requests to the endpoints.

### Text to Image Search
Use the following command to send a POST request for text-to-image search:
```bash
http POST http://127.0.0.1:8000/search-text query="A beautiful sunset over the mountains" num_images:=5
```
### Image to Image Search
To perform an image-to-image search, use the following curl command:
```bash
curl -X POST "http://127.0.0.1:8000/search-image" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/home/iot/Desktop/cat.jpg" -F "num_images=50"
```
License
This project is licensed under the MIT License - see the LICENSE file for details.


### Instructions to Update
- Replace `username` and `existing-repo` in the clone command with your actual GitHub username and repository name.
- Adjust the license section as necessary, depending on your project's licensing terms.