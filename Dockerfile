FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including git for pip
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app ./app
COPY dev ./dev

# Copy pre-downloaded models into the container
COPY model/clip /app/model/clip
COPY model/face /app/model/face

# ENV TORCHSERVE_URL=http://host.docker.internal:8080/predictions/clip
# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.api.fastapi_ai_search:app", "--host", "0.0.0.0", "--port", "8000"]