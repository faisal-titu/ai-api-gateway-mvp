# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory inside the Docker container
WORKDIR /app

RUN apt-get update && apt-get install -y git


# Step 3: Copy the requirements file into the Docker container
COPY requirements.txt .

# Step 4: Install any dependencies required by the application
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the FastAPI application code to the container
COPY . .

# Step 6: Expose the port that the FastAPI app runs on
EXPOSE 8000

# Step 7: Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "fastapi_text_search:app", "--host", "0.0.0.0", "--port", "8000"]
