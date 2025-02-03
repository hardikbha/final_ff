# ` def process_frame(frame, image_path, face_detector, face_predictor, file_extension, SAVE_IMGS_PATH, IMG_META_DICT):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (for processing)
#         faces = face_detector(frame, 1)  # Detect faces in the image
#         print("faces",len(faces))

#         if len(faces) == 0:
#             print(f'No faces detected in the image: {image_path}')
#             return

#         # Process landmarks for the detected faces
#         frame_image_dict = dict()

#         landmarks = list()  # List to store the landmarks
#         size_list = list()  # List to store the size of the detected faces

#         for face_idx in range(len(faces)):
#             landmark = face_predictor(frame, faces[face_idx])
#             landmark = face_utils.shape_to_np(landmark)

#             # Get bounding box coordinates of the face from the landmarks
#             x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
#             x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()

#             # Calculate face size (area of bounding box)
#             face_s = (x1 - x0) * (y1 - y0)
#             size_list.append(face_s)
#             landmarks.append(landmark)

#         # Select the landmark of the largest face
#         landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
#         landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

#         # Save the landmark with the biggest face
#         frame_image_dict['landmark'] = landmarks.tolist()
#         frame_image_dict['source_path'] = SAVE_IMGS_PATH
#         frame_image_dict['label'] = 0

#         # Store the metadata for this image
#         IMG_META_DICT[f"{SAVE_IMGS_PATH}"] = frame_image_dict

#         # Save the frame as a PNG image (you can save as a new processed file)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
#         image_save_path = f"{SAVE_IMGS_PATH}/processed_image{file_extension}"
#         cv2.imwrite(image_save_path, frame)
#         print(f"Processed image saved to {image_save_path}")


#     @staticmethod
#     # Function to process the image, extract facial landmarks, and save metadata
#     async def process_image(image_path, PREDICTOR_PATH, file_extension, SAVE_IMGS_PATH, IMG_META_DICT):
#         print("processing image....")
#         try:
#             # Load the image and initialize the face detector and predictor
#             frame = cv2.imread(image_path)
#             face_detector = dlib.get_frontal_face_detector()
#             face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

#             # Process the image to extract landmarks and save it
#             FileManager.process_frame(frame, image_path, face_detector, face_predictor, file_extension, SAVE_IMGS_PATH, IMG_META_DICT)

#             # Save metadata as a JSON file
#             with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
#                 json.dump(IMG_META_DICT, f)

#         except Exception as e:
#             raise custom_utils.CustomException(status_code=500, message="An error occurred while processing the image.") from e

#     @staticmethod
#     async def upload_image(file):
#         user_id = custom_utils.generate_unique_id()
#         unique_id = custom_utils.generate_unique_id()
#         upload_dir = unique_id
#         image_path = f"{upload_dir}/{unique_id}_{file.filename}"
#         file_extension = os.path.splitext(file.filename)[-1].lower()
#         valid_extensions = ['.jpg', '.png', '.jpeg']
#         if file_extension not in valid_extensions:
#             raise custom_utils.CustomException(status_code=400,
#                                                 message="Unsupported file type")
#         SAVE_IMGS_PATH = os.path.join(os.getcwd(), 'images')
#         os.makedirs(SAVE_IMGS_PATH, exist_ok=True)
#         PREDICTOR_PATH = "D:/AL-DeepFake/DeepFakeDetection/shape_predictor_81_face_landmarks.dat"
#         IMG_META_DICT = dict()
#         # Save the uploaded image locally
#         file_path = os.path.join(SAVE_IMGS_PATH, f"{file.filename}")
#         with open(file_path, 'wb') as f:
#             f.write(file.file.read())
#         await FileManager.process_image(file_path, PREDICTOR_PATH, file_extension, SAVE_IMGS_PATH, IMG_META_DICT)
#         cmd = [
#             'python', 'D:/AL-DeepFake/DeepFakeDetection/services/image_processing.py',
#             '--cfg', 'D:/AL-DeepFake/DeepFakeDetection/configs/caddm_test.cfg'
#         ]
#         try:
#             result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#             print("Evaluation script output:", result.stdout)
#         except subprocess.CalledProcessError as e:
#             print("Error running evaluation script:", e)
#             print("stderr:", e.stderr)
#             raise custom_utils.CustomException(status_code=500, message="Error running evaluation script.")
#         try:
#             # file_like_object = io.BytesIO(file) 
#             # s3_connection.s3_client.upload_fileobj(
#             #     file_like_object, Bucket=bucket_name, Key=image_path)
#             # upload_video_file.DeepFakeManager.create_file_item(
#             #     image_path, unique_id, user_id, file_extension,
#             #     file.filename)
#             return {
#                 "user_id": user_id,
#                 "file_id": unique_id,
#                 "filename": file.filename,
#                 "file_path": image_path,
#                 "message": "File Uploaded Successfully"
#             }
#         except Exception as e:
#             raise custom_utils.CustomException(
#                 status_code=500,
#                 message="An error occurred while uploading the image."
#             ) from e
# `




# # Face Crop Param
# crop_face:
#   face_width: 80
#   output_size: 224
#   scale: 0.9

# # Artifact Detection Module.
# adm_det:
#   min_dim: 224
#   aspect_ratios: [[1], [1], [1], [1]]
#   feature_maps: [7, 5, 3, 1]
#   steps: [32, 45, 75, 224]
#   min_sizes: [40, 80, 120, 224]
#   max_sizes: [80, 120, 160, 224]
#   clip: True
#   variance: [0.1]
#   name: deepfake

# # The Size of the Sliding Window.
# sliding_win:
#   prior_bbox: [[40, 80], [80, 120], [120, 160], [224, 224]]

# # test data
# dataset:
#   img_path: "./test_images"
#   ld_path: "./test_images/ldm.json"
#   name: 'FF++'


# model:
#   backbone: "resnet34"
#   ckpt: "./checkpoints/resnet34.pkl"





# #!/usr/bin/env python3
# import argparse
# from collections import OrderedDict
# from sklearn.metrics import roc_auc_score
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import cv2
# import model
# from detection_layers.modules import MultiBoxLoss
# from lib.util import load_config, update_learning_rate, my_collate, get_video_auc





# #!/usr/bin/env python3
# import os
# import cv2
# import json
# import numpy as np
# from typing import Dict, List, Tuple
# import torch
# from torch.utils.data import Dataset
# from natsort import natsorted

# #!/usr/bin/env python3
# import argparse

# import os
# import cv2
# import torch
# import random
# import numpy as np

# from functools import lru_cache
# from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage.interpolation import map_coordinates

# from .mfs import multi_scale_facial_swap
# from .augmentor import add_noise, resize_aug, image_h_mirror
# from .cropface import get_align5p, align_5p

# from detection_layers.box_utils import match
# from detection_layers import PriorBox

# Prior = None


# def get_prior(config):
#     global Prior
#     if Prior is None:
#         Prior = PriorBox(config['adm_det'])


# def label_assign(bboxs, config, genuine=False):

#     global Prior
#     get_prior(config)

#     labels = torch.zeros(bboxs.shape[0],)
#     defaults = Prior.forward().data  # information of priors

#     if genuine:
#         return np.zeros(defaults.shape), np.zeros(defaults.shape[0], )

#     # anchor matching by iou.

#     loc_t = torch.zeros(1, defaults.shape[0], 4)
#     conf_t = torch.zeros(1, defaults.shape[0])

#     match(
#         0.5, torch.Tensor(bboxs), defaults,
#         [0.1, 0.2], labels, loc_t, conf_t, 0)

#     loc_t, conf_t = np.array(loc_t)[0, ...], np.array(conf_t)[0, ...]

#     if loc_t.max() > 10**5:
#         return None, 'prior bbox match err. bias is inf!'

#     return loc_t, conf_t


# def prepare_train_input(targetRgb, sourceRgb, landmark, label, config, training=True):
#     '''Prepare model input images.

#     Arguments:
#     targetRgb: original images or fake images.
#     sourceRgb: source images.
#     landmark: face landmark.
#     label: deepfake labels. genuine: 0, fake: 1.
#     config: deepfake config dict.
#     training: return processed image with aug or not.
#     '''

#     rng = np.random

#     images = [targetRgb, sourceRgb]

#     if training and rng.rand() >= 0.7:
#         images, landmark = resize_aug(images, landmark)

#     # multi-scale facial swap.

#     targetRgb, sourceRgb = images
#     # if input image is genuine.
#     mfs_result, bbox = targetRgb, np.zeros((1, 4))
#     # if input image is fake image. generate new fake image with mfs.
#     if label:
#         blending_type = 'poisson' if rng.rand() >= 0.5 else 'alpha'

#         if rng.rand() >= 0.2:
#             # global facial swap.
#             sliding_win = targetRgb.shape[:2]

#             if rng.rand() > 0.5:
#                 # fake to source global facial swap.
#                 mfs_result, bbox = multi_scale_facial_swap(
#                     targetRgb, sourceRgb, landmark, config,
#                     sliding_win, blending_type, training
#                 )
#             elif rng.rand() >= 0.5:
#                 # source to fake global facial swap.
#                 mfs_result, bbox = multi_scale_facial_swap(
#                     sourceRgb, targetRgb, landmark, config,
#                     sliding_win, blending_type, training
#                 )
#             else:
#                 mfs_result, bbox = targetRgb, np.array([[0, 0, 224, 224]])
#                 cropMfs, landmark = get_align5p(
#                     [mfs_result], landmark, rng, config, training
#                 )
#                 mfs_result = cropMfs[0]
#         else:
#             # parial facial swap.
#             prior_bbox = config['sliding_win']['prior_bbox']
#             sliding_win = prior_bbox[np.random.choice(len(prior_bbox))]
#             mfs_result, bbox = multi_scale_facial_swap(
#                 sourceRgb, targetRgb, landmark, config,
#                 sliding_win, blending_type, training
#             )
#     else:
#         # crop face with landmark.
#         cropMfs, landmark = get_align5p(
#             [mfs_result], landmark, rng, config, training
#         )
#         mfs_result = cropMfs[0]

#     if mfs_result is None:
#         return None, 'multi scale facial swap err.'

#     if training:  # and rng.rand() >= 0.5:
#         mfs_result, bbox = image_h_mirror(mfs_result, bbox)
#         mfs_result = add_noise(rng, mfs_result)

#     genuine = True if not label else False

#     location_label, confidence_label = label_assign(
#         bbox.astype('float32') / config['crop_face']['output_size'],
#         config, genuine
#     )

#     return mfs_result, {'label': label, 'location_label': location_label,
#                         'confidence_label': confidence_label}


# def prepare_test_input(img, ld, label, config):
#     config = config['crop_face']

#     img, ld = align_5p(
#         img, ld=ld,
#         face_width=config['face_width'], canvas_size=config['output_size'],
#         scale=config['scale']
#     )
#     return img, {'label': label}

# # vim: ts=4 sw=4 sts=4 expandtab


# class DeepfakeDataset(Dataset):
#     r"""DeepfakeDataset Dataset.

#     The folder is expected to be organized as followed: root/cls/xxx.img_ext

#     Labels are indices of sorted classes in the root directory.

#     Args:
#         mode: train or test.
#         config: hypter parameters for processing images.
#     """

#     def __init__(self, mode: str, config: dict):
#         super().__init__()

#         self.config = config
#         self.mode = mode
#         self.root = self.config['dataset']['img_path']
#         self.landmark_path = self.config['dataset']['ld_path']
#         self.rng = np.random
#         assert mode in ['train', 'test']
#         self.do_train = True if mode == 'train' else False
#         self.info_meta_dict = self.load_landmark_json(self.landmark_path)
#         self.class_dict = self.collect_class()
#         self.samples = self.collect_samples()

#     def load_landmark_json(self, landmark_json) -> Dict:
#         with open(landmark_json, 'r') as f:
#             landmark_dict = json.load(f)
#         return landmark_dict

#     def collect_samples(self) -> List:
#         samples = []
#         directory = os.path.expanduser(self.root)
#         for key in sorted(self.class_dict.keys()):
#             d = os.path.join(directory, key)
#             print("d", d)
#             if not os.path.isdir(d):
#                 continue
#             for r, _, filename in sorted(os.walk(d, followlinks=True)):
#                 print("r", r)
#                 print("f_name", filename)
#                 for name in natsorted(filename):
#                     print("name", name)
#                     if name.startswith('.'):  # Skip hidden/system files
#                         continue
#                     path = os.path.join(r, name)
#                     print("path", path)
#                     info_key = path[:-4]  # Remove file extension for the key
#                     video_name = '/'.join(path.split('/')[:-1])
#                     print("V_name", video_name)

#                     # Check if the key exists in the metadata dictionary
#                     if info_key not in self.info_meta_dict:
#                         print(f"Warning: {info_key} not found in info_meta_dict. Skipping...")
#                         continue

#                     # Access metadata and construct the sample
#                     info_meta = self.info_meta_dict[info_key]
#                     landmark = info_meta['landmark']
#                     class_label = int(info_meta['label'])
#                     source_path = info_meta['source_path'] + path[-4:]
#                     samples.append(
#                         (path, {'labels': class_label, 'landmark': landmark,
#                                 'source_path': source_path,
#                                 'video_name': video_name})
#                     )

#         return samples


#     def collect_class(self) -> Dict:
#         classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
#         # classes.sort(reverse=True)
#         return {classes[i]: np.int32(i) for i in range(len(classes))}

#     def __getitem__(self, index: int) -> Tuple:
#         path, label_meta = self.samples[index]
#         ld = np.array(label_meta['landmark'])
#         label = label_meta['labels']
#         source_path = label_meta['source_path']
#         img = cv2.imread(path, cv2.IMREAD_COLOR)


#         img_frame = img
#         source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)


#         if self.mode == "train":
#             img, label_dict = prepare_train_input(
#                 img, source_img, ld, label, self.config, self.do_train
#             )
#             if isinstance(label_dict, str):
#                 return None, label_dict

#             location_label = torch.Tensor(label_dict['location_label'])
#             confidence_label = torch.Tensor(label_dict['confidence_label'])
#             img = torch.Tensor(img.transpose(2, 0, 1))
#             return img, (label, location_label, confidence_label)

#         elif self.mode == 'test':
#             img, label_dict = prepare_test_input([img], ld, label, self.config)

#             print(f"Accessing index: {index}")

#             img1 = img[0]
#             # print("img_shape_2", im)

#             # Save the image using OpenCV in PNG format
#             output_path = 'save_image.png'
#             cv2.imwrite(output_path, img1)


#             # Convert tensor (C, H, W) to numpy array (H, W, C)

#             img = torch.Tensor(img[0].transpose(2, 0, 1))

#             # Retrieve video name
#             video_name = label_meta['video_name']

#             print("IMf", img_frame.shape)

#             return img, (label, video_name), img_frame, ld        

#         else:
#             raise ValueError("Unsupported mode of dataset!")

#     def __len__(self):
#         return len(self.samples)


# if __name__ == "__main__":
#     from lib.util import load_config
#     config = load_config('./configs/caddm_train.cfg')
#     d = DeepfakeDataset(mode="test", config=config)
#     for index in range(len(d)):
#         res = d[index]
# # vim: ts=4 sw=4 sts=4 expandtab











# def args_func():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
#     args = parser.parse_args()
#     return args


# def load_checkpoint(ckpt, net, device):
#     checkpoint = torch.load(ckpt)

#     gpu_state_dict = OrderedDict()
#     for k, v in checkpoint['network'] .items():
#         name = "module." + k  # add `module.` prefix
#         gpu_state_dict[name] = v.to(device)
#     net.load_state_dict(gpu_state_dict)
#     return net


# def draw_landmarks_and_bbox_batch(images, landmarks_batch, predicted_labels):
#     """
#     Draw landmarks on each image in the batch, calculate a bounding box around the landmarks,
#     save the frames in an organized folder structure, and create an MP4 video.

#     Args:
#     - images: The input batch of images (tensor of shape [batch_size, height, width, channels]).
#     - landmarks_batch: A list of lists, where each inner list contains tuples of landmarks for one image.
#                         Example: [[(x1, y1), (x2, y2)], [(x1, y1), (x2, y2)], ...]
#     - predicted_labels: A list or tensor containing label scores for the images.

#     Returns:
#     - The batch of images with landmarks and bounding boxes drawn.
#     """
#     batch_size, height, width, channels = images.shape

#     # Create an empty list to store the images with bounding boxes
#     output_images = []

#     # Create directories for saving frames and video
#     os.makedirs("output/frames", exist_ok=True)
#     os.makedirs("output/video", exist_ok=True)

#     # Video writer setup
#     video_path = os.path.join("output/video", "output_video.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
#     video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

#     # Process each image in the batch
#     for i in range(batch_size):
#         print("b_s", batch_size)
#         image = images[i]  # Get image i from the batch
#         print("I_SH", image.shape)

#         image = image.detach().cpu().numpy()
#         image = (image - image.min()) / (image.max() - image.min())  # Scale to [0, 1]
#         image = (image * 255).astype(np.uint8)  # Convert to [0, 255] and ensure uint8
#         print("im_shape", image.shape)

#         labelsS = predicted_labels[i] if isinstance(predicted_labels, (list, torch.Tensor)) else predicted_labels

#         print("P_Labels", labelsS)
#         print("dt_labels", type(labelsS))

#         # Get the landmarks for this image
#         landmarks = landmarks_batch[i]
#         print("landmarks", landmarks)


#         # Calculate the bounding box (min and max x and y coordinates)
#         xs = [int(x) for x, y in landmarks]
#         ys = [int(y) for x, y in landmarks]

#         x_min = min(xs)
#         y_min = min(ys)
#         x_max = max(xs)
#         y_max = max(ys)

#         # Create a consistent bounding box for the entire video
#         if (labelsS > 0.40):
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box (Fake)
#             label_text1 = "Fake"
#             text_position = (x_min, y_min - 10)  # Place text slightly above the top-left corner of the box
#             cv2.putText(
#                 image,
#                 label_text1,
#                 text_position,
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=0.5,
#                 color=(0, 0, 255),  # Red text
#                 thickness=1,
#                 lineType=cv2.LINE_AA
#             )
#         else:
#             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box (Real)
#             label_text2 = "Real"
#             text_position = (x_min, y_min - 10)  # Place text slightly above the top-left corner of the box
#             cv2.putText(
#                 image,
#                 label_text2,
#                 text_position,
#                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                 fontScale=0.5,
#                 color=(0, 255, 0),  # Green text
#                 thickness=1,
#                 lineType=cv2.LINE_AA
#             )

#         # Save each frame as an individual image in the frames folder
#         frame_filename = os.path.join("output/frames", f'frame_{i:04d}.png')
#         cv2.imwrite(frame_filename, image)

#         # Add frame to video writer
#         video_writer.write(image)

#         # Append to the output_images list
#         output_images.append(image)

#     # Release the video writer
#     video_writer.release()

#     return np.array(output_images)


# def test():
#     args = args_func()

#     # load conifigs
#     cfg = load_config(args.cfg)

#     # init model.
#     net = model.get(backbone=cfg['model']['backbone'])
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
#     net = nn.DataParallel(net)
#     net.eval()
#     if cfg['model']['ckpt']:
#         net = load_checkpoint(cfg['model']['ckpt'], net, device)

#     # get testing data
#     print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
#     test_dataset = DeepfakeDataset('test', cfg)
#     test_loader = DataLoader(test_dataset,
#                              batch_size=cfg['test']['batch_size'],
#                              shuffle=False, num_workers=0,
#                              )

#     # start testing.
#     frame_pred_list = list()
#     frame_label_list = list()
#     video_name_list = list()

#     for batch_data, batch_labels, img_frame, ld in test_loader:

#         labels, video_name = batch_labels
#         labels = labels.long()

#         outputs = net(batch_data)
#         outputs = outputs[:, 1]
#         outputs = outputs.detach().cpu().numpy().tolist()

#         output_image = draw_landmarks_and_bbox_batch(img_frame, ld, outputs)


# if __name__ == "__main__":
#     test()

# # vim: ts=4 sw=4 sts=4 expandtab























from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import dlib
from imutils import face_utils
from collections import OrderedDict
import subprocess




app = FastAPI(title="DeepFake Detection API")

class DeepFakeDetector:
    def __init__(self, config_path: str, model_path: str, predictor_path: str):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.load_model(model_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(predictor_path)
        
    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_model(self, model_path: str):
        # Initialize your model here based on config
        net = self.get_model(backbone=self.config['model']['backbone'])
        net = net.to(self.device)
        net = nn.DataParallel(net)
        net.eval()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        gpu_state_dict = OrderedDict()
        for k, v in checkpoint['network'].items():
            name = "module." + k
            gpu_state_dict[name] = v.to(self.device)
        net.load_state_dict(gpu_state_dict)
        return net

    def process_frame(self, frame: np.ndarray):
        """Process a single frame for deepfake detection"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(frame_rgb, 1)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No faces detected in the image")
            
        landmarks_list = []
        size_list = []
        
        for face in faces:
            landmark = self.face_predictor(frame_rgb, face)
            landmark = face_utils.shape_to_np(landmark)
            
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_size = (x1 - x0) * (y1 - y0)
            
            landmarks_list.append(landmark)
            size_list.append(face_size)
            
        # Get largest face landmarks
        landmarks = landmarks_list[np.argmax(size_list)]
        
        # Process image according to model requirements
        processed_img = self.prepare_image_for_model(frame, landmarks)
        return processed_img, landmarks

    def detect(self, image: np.ndarray) -> Dict:
        """Perform deepfake detection on an image"""
        try:
            processed_img, landmarks = self.process_frame(image)
            
            # Convert to tensor and move to device
            img_tensor = torch.from_numpy(processed_img).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.net(img_tensor)
                scores = torch.softmax(outputs, dim=1)
                fake_score = scores[0][1].item()
            
            # Draw detection results
            result_image = self.draw_detection_results(image, landmarks, fake_score)
            
            return {
                "is_fake": fake_score > 0.5,
                "fake_probability": fake_score,
                "detection_image": result_image
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def draw_detection_results(self, image: np.ndarray, landmarks: np.ndarray, fake_score: float) -> np.ndarray:
        """Draw detection results on the image"""
        result = image.copy()
        
        # Draw bounding box
        x_min, y_min = landmarks[:, 0].min(), landmarks[:, 1].min()
        x_max, y_max = landmarks[:, 0].max(), landmarks[:, 1].max()
        
        color = (0, 0, 255) if fake_score > 0.5 else (0, 255, 0)
        label = f"Fake: {fake_score:.2%}" if fake_score > 0.5 else f"Real: {1-fake_score:.2%}"
        
        cv2.rectangle(result, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(result, label, (int(x_min), int(y_min-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result

# Initialize detector
detector = DeepFakeDetector(
    config_path="configs/caddm_test.cfg",
    model_path="checkpoints/resnet34.pkl",
    predictor_path="shape_predictor_81_face_landmarks.dat"
)

@app.post("/detect/")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Endpoint to detect if an image is deepfake
    """
    try:
        # Validate file extension
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform detection
        result = detector.detect(image)
        
        # Save result image
        output_path = f"output/{file.filename}"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, result["detection_image"])
        
        return {
            "filename": file.filename,
            "is_fake": result["is_fake"],
            "confidence": result["fake_probability"],
            "result_image_path": output_path,
            "message": "Detection completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)