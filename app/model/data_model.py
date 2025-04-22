from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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
    num_images: int

# Endpoint to dynamically set index name and image directory
class SettingsRequest(BaseModel):
    index_name: str
    image_dir: str

class ProcessedImage(BaseModel):
    bounding_box: List[int]
    confidence: float
    cropped_width: int
    cropped_height: int
    cropped_resolution: str
    cropped_image_path: str

class FaceEmbedding(BaseModel):
    # Adjust the fields based on your actual embedding structure
    face_id: str
    box: List[int]
    embedding: List[float] 

class FaceDetectionResult(BaseModel):
    status: str
    image_name: str
    total_faces_detected: int
    processed_images: List[ProcessedImage]
    embeddings: List[FaceEmbedding]

class IndexingDetail(BaseModel):
    image_name: str
    face_id: Optional[str]
    status: str  # e.g., "indexed" or "error"
    error_message: Optional[str] = None

class BatchEmbeddingDetails(BaseModel):
    total_images_processed: int
    total_faces_detected: int
    total_embeddings_indexed: int
    indexing_details: List[IndexingDetail]

class BatchFaceEmbeddingResult(BaseModel):
    status: str
    face_crop_dir: str
    details: BatchEmbeddingDetails

class CombinedSearchDetail(BaseModel):
    image_search: Optional[ImageSearchResult]
    face_search: Optional[Dict[str, Any]]

class CombinedSearchResponse(BaseModel):
    status: str
    details: CombinedSearchDetail
    error: Optional[str] = None