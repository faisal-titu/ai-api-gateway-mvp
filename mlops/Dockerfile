# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Copy the requirements file into the Docker container
COPY ../requirements.txt .

# Install any dependencies required by the application
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code to the container
COPY ../app ./app

# Copy the dev directory to the container
COPY ../dev ./dev

# Expose the port that the FastAPI app runs on
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.api.fastapi_ai_search:app", "--host", "0.0.0.0", "--port", "8000"]