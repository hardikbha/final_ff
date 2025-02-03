import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch.utils.data import DataLoader
import model
from dataset import DeepfakeDataset
from lib.util import load_config
from collections import OrderedDict
import tempfile
import shutil

# Setup directories
OUTPUT_DIR = "output"
os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "video"), exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Initialize model and device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model.get(backbone='resnet')  # Use the backbone specified in your model's config
net = nn.DataParallel(net).to(device)
net.eval()

def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)
    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'].items():
        name = "module." + k  # Add `module.` prefix for multi-GPU setup
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net

# Load model checkpoint
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)


# Define the response model for API output
class APIResponse(BaseModel):
    processed_image_path: str
    metadata: dict

def draw_landmarks_and_bbox_batch(images, landmarks_batch, predicted_labels):
    batch_size, height, width, channels = images.shape
    output_images = []

    # Video writer setup
    video_path = os.path.join(OUTPUT_DIR, "video", "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    # Process each image in the batch
    for i in range(batch_size):
        image = images[i].detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Scale to [0, 1]
        image = (image * 255).astype(np.uint8)  # Convert to [0, 255] and ensure uint8

        labels = predicted_labels[i] if isinstance(predicted_labels, (list, torch.Tensor)) else predicted_labels
        landmarks = landmarks_batch[i]

        # Calculate the bounding box
        xs = [int(x) for x, y in landmarks]
        ys = [int(y) for x, y in landmarks]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        # Draw bounding box and label
        if (labels > 0.40):
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box (Fake)
            label_text = "Fake"
            text_position = (x_min, y_min - 10)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box (Real)
            label_text = "Real"
            text_position = (x_min, y_min - 10)

        cv2.putText(image, label_text, text_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        # Save frames
        frame_filename = os.path.join(OUTPUT_DIR, "frames", f'frame_{i:04d}.png')
        cv2.imwrite(frame_filename, image)

        # Add frame to video writer
        video_writer.write(image)

        output_images.append(image)

    # Release the video writer
    video_writer.release()

    return np.array(output_images), frame_filename

@app.post("/process_image/", response_model=APIResponse)
async def process_image(file: UploadFile = File(...)):
    # Create a temporary directory for image processing
    temp_dir = tempfile.mkdtemp()

    # Save the uploaded file to the temp directory
    img_path = os.path.join(temp_dir, "uploaded_image.jpg")
    with open(img_path, "wb") as img_file:
        shutil.copyfileobj(file.file, img_file)

    # Read the uploaded image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess image and make prediction
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = net(img_tensor)
        outputs = outputs[:, 1]  # Assuming binary classification (Real vs Fake)
        landmarks = []  # Replace with your own method to extract landmarks if necessary
        predicted_labels = outputs.cpu().numpy().tolist()

    # Draw landmarks and bounding boxes on the image
    processed_image, frame_filename = draw_landmarks_and_bbox_batch(img_tensor, landmarks, predicted_labels)

    return APIResponse(processed_image_path=frame_filename, metadata={"predicted_labels": predicted_labels})

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Detection API"}

# Run the API with:
# uvicorn app_name:app --reload
