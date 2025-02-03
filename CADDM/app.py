from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import shutil
import tempfile
import json
from typing import Dict

# Setup directories and paths
VIDEO_PATH = "./data/FaceForensics++"
SAVE_IMGS_PATH = "./test_images"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
IMG_META_DICT = dict()

# Initialize FastAPI app
app = FastAPI()

# Load face detector and predictor
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

def parse_labels(image_path: str):
    label = 0 if "original" in image_path else 1
    return label

def process_frame(frame, cnt_frame, save_path, label, face_detector, face_predictor):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(frame, 1)

    if len(faces) == 0:
        return None, 'No faces detected'

    landmarks = list()
    size_list = list()

    for face_idx in range(len(faces)):
        landmark = face_predictor(frame, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks.append(landmark)

    landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

    # save the meta info of the current frame
    frame_image_dict = {
        'landmark': landmarks.tolist(),
        'label': label
    }

    # Store frame-specific metadata
    IMG_META_DICT[f"frame_{cnt_frame}"] = frame_image_dict

    # Save the processed frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    temp_filename = f"/tmp/frame_{cnt_frame}.png"
    cv2.imwrite(temp_filename, frame)

    return temp_filename, frame_image_dict

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Create a temporary directory for image processing
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to temp directory
    img_path = os.path.join(temp_dir, "uploaded_image.jpg")
    with open(img_path, "wb") as img_file:
        shutil.copyfileobj(file.file, img_file)

    # Read the uploaded image
    image = cv2.imread(img_path)

    # Process the frame
    processed_image_path, frame_metadata = process_frame(image, 0, temp_dir, parse_labels(img_path), face_detector, face_predictor)

    if processed_image_path is None:
        return JSONResponse(content={"error": frame_metadata}, status_code=400)

    # Return metadata
    return JSONResponse(content={
        "metadata": frame_metadata,
        "processed_image_path": processed_image_path
    })

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Landmark Detection API"}

# Run the API with:
# uvicorn app_name:app --reload
