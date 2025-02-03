# import argparse
# from glob import glob
# import os
# import cv2
# import dlib
# import json
# import numpy as np
# from tqdm import tqdm
# from imutils import face_utils
# from sklearn.metrics import roc_auc_score
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import model
# from detection_layers.modules import MultiBoxLoss
# from dataset import DeepfakeDataset
# from lib.util import load_config, update_learning_rate, my_collate, get_video_auc

# VIDEO_PATH = "./data/FaceForensics++"
# SAVE_IMGS_PATH = "./test_images"
# PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
# DATASETS = {'Original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes', 'NeuralTextures'}
# COMPRESSION = {'raw'}
# NUM_FRAMES = 1
# IMG_META_DICT = dict()

# def parse_video_path(dataset, compression):
#     if dataset == 'Original':
#         dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
#     elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
#         dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
#     else:
#         raise NotImplementedError

#     movies_path_list = sorted(glob(dataset_path + '*.mp4'))
#     print(f"{len(movies_path_list)} : videos are exist in {dataset}")
#     return movies_path_list

# def parse_labels(video_path):
#     return 0 if "original" in video_path else 1

# def parse_source_save_path(save_path):
#     if "original" in save_path:
#         return save_path

#     img_meta = save_path.split('/')
#     source_target_index = img_meta[-1]
#     manipulation_name = img_meta[-4]
#     original_name = "youtube"
#     source_index = img_meta[-2]  # Adjust based on your directory structure.

#     return save_path.replace("manipulated_sequences", "original_sequences")\
#                     .replace(manipulation_name, original_name)\
#                     .replace(source_target_index, source_index)

# def test():
#     args = args_func()
#     cfg = load_config(args.cfg)
#     net = model.get(backbone=cfg['model']['backbone'])
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
#     net = nn.DataParallel(net)
#     net.eval()

#     if cfg['model']['ckpt']:
#         checkpoint = torch.load(cfg['model']['ckpt'])
#         net.load_state_dict({"module." + k: v.to(device) for k, v in checkpoint['network'].items()})

#     print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
#     test_dataset = DeepfakeDataset('test', cfg)
#     test_loader = DataLoader(test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=0)

#     frame_pred_list, frame_label_list, video_name_list = [], [], []

#     for batch_data, batch_labels in test_loader:
#         labels, video_name = batch_labels
#         labels = labels.long()

#         outputs = net(batch_data)
#         outputs = outputs[:, 1]  # Extract "Fake" probabilities

#         # Debug: Log predictions and labels
#         print(f"Batch outputs: {outputs.detach().cpu().numpy()}")
#         print(f"Batch labels: {labels.detach().cpu().numpy()}")

#         frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
#         frame_label_list.extend(labels.detach().cpu().numpy().tolist())
#         video_name_list.extend(list(video_name))

#     f_auc = roc_auc_score(frame_label_list, frame_pred_list)
#     v_auc = get_video_auc(frame_label_list, video_name_list, frame_pred_list)

#     print(f"Frame-AUC: {f_auc:.4f}")
#     print(f"Video-AUC: {v_auc:.4f}")
# def preprocess_video(video_path, save_path, face_detector, face_predictor, model, device):
#     video_dict = dict()
#     label = parse_labels(video_path)
#     source_save_path = parse_source_save_path(save_path)
#     os.makedirs(save_path, exist_ok=True)
#     cap_video = cv2.VideoCapture(video_path)
#     frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=np.int)

#     processed_frames = []
    
#     for cnt_frame in range(frame_count_video):
#         ret, frame = cap_video.read()
#         if not ret:
#             tqdm.write(f'Frame read {cnt_frame} Error! : {os.path.basename(video_path)}')
#             continue
#         if cnt_frame not in frame_idxs:
#             continue

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_detector(frame_rgb, 1)
#         if len(faces) == 0:
#             tqdm.write(f'No faces in {cnt_frame}:{os.path.basename(video_path)}')
#             continue

#         landmarks = list()
#         size_list = list()
#         for face_idx in range(len(faces)):
#             landmark = face_predictor(frame_rgb, faces[face_idx])
#             landmark = face_utils.shape_to_np(landmark)
#             x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
#             x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
#             face_s = (x1 - x0) * (y1 - y0)
#             size_list.append(face_s)
#             landmarks.append(landmark)

#         # Select the largest face
#         landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
#         landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

#         # Resize for model input, but keep original image for visualization
#         full_frame_resized = cv2.resize(frame_rgb, (224, 224))
#         input_tensor = torch.from_numpy(full_frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
#         processed_frames.append((input_tensor, frame, x0, y0, x1, y1, cnt_frame, landmarks))

#     # Batch process the frames
#     if processed_frames:
#         input_tensors = torch.cat([frame[0] for frame in processed_frames])
        
#         model.eval()
#         with torch.no_grad():
#             outputs = model(input_tensors)
            
#             # Print batch outputs exactly as in the test function
#             print(f"Batch outputs: {outputs.detach().cpu().numpy()}")
            
#             confidence_scores = outputs[:, 1].cpu().numpy()

#         # Process each frame with its corresponding confidence score
#         for (input_tensor, original_frame, x0, y0, x1, y1, cnt_frame, landmarks) in processed_frames:
#             confidence_score = float(confidence_scores[processed_frames.index((input_tensor, original_frame, x0, y0, x1, y1, cnt_frame, landmarks))])
            
#             # Determine label based on batch output
#             predicted_label = "Fake" if confidence_score > 0.5 else "Original"

#             video_dict['landmark'] = landmarks.tolist()
#             video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
#             video_dict['label'] = label
#             video_dict['confidence_score'] = confidence_score
#             video_dict['predicted_label'] = predicted_label
#             IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = video_dict

#             color = (0, 0, 255) if predicted_label == "Fake" else (0, 255, 0)
#             label_text = f"{predicted_label} ({confidence_score:.4f})"
            
#             # Draw bounding box on the original frame
#             frame_with_box = original_frame.copy()
#             cv2.rectangle(frame_with_box, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
#             cv2.putText(frame_with_box, label_text, (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
#             image_path = f"{save_path}/frame_{cnt_frame}.png"
            
#             # Ensure directory exists
#             os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
#             cv2.imwrite(image_path, frame_with_box)

#     cap_video.release()

# def args_func():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
#     return parser.parse_args()

# if __name__ == '__main__':
#     face_detector = dlib.get_frontal_face_detector()
#     face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
#     # import ipdb; ipdb.set_trace()

#     cfg = load_config('./configs/caddm_test.cfg')
#     net = model.get(backbone=cfg['model']['backbone'])
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
#     net.load_state_dict(torch.load(cfg['model']['ckpt'], map_location=device)['network'])
#     net.eval()

#     for dataset in DATASETS:
#         for comp in COMPRESSION:
#             movies_path_list = parse_video_path(dataset, comp)
#             for video_path in tqdm(movies_path_list):
#                 save_path_per_video = video_path.replace(VIDEO_PATH, SAVE_IMGS_PATH).replace('.mp4', '').replace("/videos", "/frames")
#                 preprocess_video(video_path, save_path_per_video, face_detector, face_predictor, net, device)

#     with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
#         json.dump(IMG_META_DICT, f)

#     test()




