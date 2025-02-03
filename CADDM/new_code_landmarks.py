#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import dlib
from imutils import face_utils
from glob import glob
import json
from collections import OrderedDict
import model  # Ensure the correct import for your model
from lib.util import load_config

VIDEO_PATH = "./data/FaceForensics++"
SAVE_IMGS_PATH = "./test_images"
ANNOTATED_FRAMES_PATH = "./annotated_frames"  # Path for annotated frames
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
DATASETS = {'Original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes', 'NeuralTextures'}
COMPRESSION = {'raw'}
NUM_FRAMES = 30
IMG_META_DICT = dict()

# Function to load model and weights
def load_model(cfg):
    # Initialize model using the configuration
    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)
    return net, device

# Function to parse video path
def parse_video_path(dataset, compression):
    if dataset == 'Original':
        dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
    elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
    else:
        raise NotImplementedError
    movies_path_list = sorted(glob(dataset_path + '*.mp4'))
    print(f"{len(movies_path_list)} videos are present in {dataset}")
    return movies_path_list

def parse_labels(video_path):
    return 0 if "original" in video_path else 1

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    args = parser.parse_args()
    return args

def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'].items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def parse_source_save_path(save_path):
    if "original" in save_path:
        return save_path
    img_meta = save_path.split('/')
    source_target_index = img_meta[-1]
    source_index = source_target_index.split('_')[0]
    manipulation_name = img_meta[-4]
    original_name = "youtube"
    return save_path.replace("manipulated_sequences", "original_sequences").replace(manipulation_name, original_name).replace(source_target_index, source_index)

def preprocess_video(video_path, save_path, annotated_save_path, face_detector, face_predictor, model, device, cfg):
    video_dict = dict()
    label = parse_labels(video_path)
    source_save_path = parse_source_save_path(save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(annotated_save_path, exist_ok=True)

    cap_video = cv2.VideoCapture(video_path)
    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=int)

    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()
        height, width = frame.shape[:-1]
        if not ret:
            tqdm.write(f'Frame read {cnt_frame} Error! : {os.path.basename(video_path)}')
            continue
        if cnt_frame not in frame_idxs:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write(f'No faces in {cnt_frame}:{os.path.basename(video_path)}')
            continue
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

        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
        video_dict['landmark'] = landmarks.tolist()
        video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
        video_dict['label'] = label
        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict

        # Model inference (using the actual model here)
        inputs = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 255.0  # Normalization for model input
        inputs = inputs.to(device)
        outputs = model(inputs)
        pred = outputs[:, 1].item()  # Assuming output class index 1 is for 'fake' (1 = fake, 0 = real)

        # Print the prediction score for debugging
        print(f"Prediction score for frame {cnt_frame} in {os.path.basename(video_path)}: {pred}")

        # Draw bounding boxes and add labels
        color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)  # Green for Real, Red for Fake
        label_text = "Fake" if pred > 0.5 else "Real"

        # Draw the rectangular bounding box around the face
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        # Annotate the frame with label
        cv2.putText(frame, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save the original frame and the annotated frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
        annotated_frame_path = f"{annotated_save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(annotated_frame_path, frame)

    cap_video.release()

def main():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Load configuration and model
    args = args_func()
    cfg = load_config(args.cfg)
    model, device = load_model(cfg)

    for dataset in DATASETS:
        for comp in COMPRESSION:
            movies_path_list = parse_video_path(dataset, comp)
            for video_path in tqdm(movies_path_list):
                save_path_per_video = video_path.replace(VIDEO_PATH, SAVE_IMGS_PATH).replace('.mp4', '').replace("/videos", "/frames")
                annotated_save_path_per_video = save_path_per_video.replace(SAVE_IMGS_PATH, ANNOTATED_FRAMES_PATH)
                preprocess_video(video_path, save_path_per_video, annotated_save_path_per_video, face_detector, face_predictor, model, device, cfg)

    with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)

if __name__ == '__main__':
    main()