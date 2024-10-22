# Face Detection and Cropping with FaceNet and MTCNN

This project performs face detection on a dataset of images, crops the detected faces, and saves them to a new directory. It uses the MTCNN model for face detection and the FaceNet InceptionResnetV1 model for potential embedding extraction. The cropped face images are saved with unique IDs, and their associated bounding boxes, face IDs, and image IDs can be extracted for further processing.

## Features

- **Face Detection**: Detects faces in images using the MTCNN model.
- **Face Cropping**: Crops the detected faces with adjustable margins around the face.
- **Unique Face ID Generation**: Assigns each cropped face a unique ID.
- **Image Processing with Progress Bar**: Uses `tqdm` for progress reporting while processing the images.
- **Face Count**: After processing, the number of detected and cropped faces in the dataset is printed.

## Prerequisites

Ensure that you have the following installed:

- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended for faster face detection)

## Installation

### Clone the repository

```bash
git clone https://github.com/faisal-titu/facenet.git
cd facenet
```

### Install the required packages

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

### Dataset Setup

- Place your dataset of images in the directory specified by `DATASET_DIR` in the code (or modify the path to your dataset).
- Ensure that the dataset contains only images in supported formats (JPEG, PNG, etc.).

## How to Run

1. Set the dataset directory path in the `DATASET_DIR` variable.
2. Set the output directory for cropped faces in the `FACE_CROP_DIR` variable.
3. Run the notebook cell by cell and you will see the output.


### Output

- Detected faces are saved in the specified `FACE_CROP_DIR` with unique IDs.
- Bounding box information for each detected face is available within the script for further usage or indexing.

### Example Code

```python
def detect_and_plot_faces(image_path, crop_folder, margin=25):
    # Face detection and cropping logic
```

### Count Detected Faces

At the end of the script, it prints the number of faces detected and saved to the `FACE_CROP_DIR`:

```python
files = [f for f in os.scandir(FACE_CROP_DIR) if f.is_file()]
print(f"Found {len(files)} faces in the dataset directory")
```

## Customization

- **Margin around faces**: You can adjust the `margin` parameter in the `detect_and_plot_faces` function to control how much space around the detected face should be included in the cropped image.
- **Confidence threshold**: The confidence threshold for face detection is set to 0.98. Modify this value in the `detect_and_plot_faces` function to make the face detection more or less sensitive.
- **GPU usage**: By default, the script will use a GPU if available. You can change this by setting `device = torch.device('cpu')` to use the CPU instead.

## Future Enhancements

- **Face Embedding**: Integrating the FaceNet model (`InceptionResnetV1`) to generate embeddings for each cropped face.
- **OpenSearch Integration**: Indexing the cropped faces and their embeddings into OpenSearch for efficient retrieval and face recognition.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## `requirements.txt` File

Here is the `requirements.txt` file for this project:

```
torch>=2.0.0
torchvision>=0.15.0
facenet-pytorch>=2.5.2
tqdm>=4.62.3
opensearch-py>=2.0.0
Pillow>=9.0.0
matplotlib>=3.5.0
uuid>=1.30
```

This will ensure that anyone can set up the environment and run your project by simply installing the required packages.