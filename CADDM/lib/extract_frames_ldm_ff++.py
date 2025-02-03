#!/usr/bin/env python3
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import dlib
import json
import argparse
from imutils import face_utils


VIDEO_PATH = "./data/FaceForensics++"
SAVE_IMGS_PATH = "./test_images"
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
DATASETS = {'Original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes', 'NeuralTextures'}
COMPRESSION = {'raw'}
NUM_FRAMES = 1000
IMG_META_DICT = dict()


def parse_video_path(dataset, compression):
    # this path setting follows FF++ dataset
    if dataset == 'Original':
        dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
    elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
    else:
        raise NotImplementedError
    # get all videos under the specific manipulated/original sequences
    movies_path_list = sorted(glob(dataset_path+'*.mp4'))
    print("{} : videos are exist in {}".format(len(movies_path_list), dataset))
    return movies_path_list


def parse_labels(video_path):
    label = None
    if "original" in video_path:
        label = 0
    else:
        label = 1
    return label


def parse_source_save_path(save_path):
    source_save_path = None
    if "original" in save_path:
        source_save_path = save_path
    else:
        img_meta = save_path.split('/')
        source_target_index = img_meta[-1]
        source_index = source_target_index.split('_')[0]
        manipulation_name = img_meta[-4]
        original_name = "youtube"
        source_save_path = save_path.replace(
            "manipulated_sequences", "original_sequences"
        ).replace(
            manipulation_name, original_name
        ).replace(
            source_target_index, source_index
        )
    return source_save_path

def preprocess_video(video_path, save_path, face_detector, face_predictor):
    # save the video meta info here
    video_dict = dict()
    # get the labels
    label = parse_labels(video_path)
    # get the path of corresponding source imgs
    source_save_path = parse_source_save_path(save_path)
    # prepare the save path
    os.makedirs(save_path, exist_ok=True)
    # read the video and prepare the sampled index
    cap_video = cv2.VideoCapture(video_path)
    frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("F_CV", frame_count_video)
    frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=int)
    
    # process each frame
    for cnt_frame in range(frame_count_video):
        ret, frame = cap_video.read()
        if not ret:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        
        if cnt_frame not in frame_idxs:
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector(frame, 1)
        
        if len(faces) == 0:
            tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(video_path)))
            continue
        
        # Process landmarks for the current frame
        frame_video_dict = dict()
        
        landmarks = list()  # save the landmark
        size_list = list()  # save the size of the detected face
        
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        
        # save the landmark with the biggest face
        landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
        
        # save the meta info of the current frame
        frame_video_dict['landmark'] = landmarks.tolist()
        frame_video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
        frame_video_dict['label'] = label
        
        # Store frame-specific metadata
        IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = frame_video_dict
        
        # save the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_path = f"{save_path}/frame_{cnt_frame}.png"
        cv2.imwrite(image_path, frame)
    
    cap_video.release()
    return


def main():
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    for dataset in DATASETS:
        for comp in COMPRESSION:
            movies_path_list = parse_video_path(dataset, comp)
            n_sample = len(movies_path_list)
            for i in tqdm(range(n_sample)):
                save_path_per_video = movies_path_list[i].replace(
                    VIDEO_PATH, SAVE_IMGS_PATH
                ).replace('.mp4', '').replace("/videos", "/frames")
                preprocess_video(
                    movies_path_list[i], save_path_per_video,
                    face_detector, face_predictor
                )
    with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
        json.dump(IMG_META_DICT, f)


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab







# #!/usr/bin/env python3
# import cv2
# import os
# import json
# import argparse
# import numpy as np
# import dlib
# from glob import glob
# from tqdm import tqdm
# from imutils import face_utils

# VIDEO_PATH = "./data/FaceForensics++"
# SAVE_IMGS_PATH = "./test_images"
# PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
# DATASETS = {'Original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes', 'NeuralTextures'}
# COMPRESSION = {'raw'}
# NUM_FRAMES = 1000
# IMG_META_DICT = dict()

# def parse_video_path(dataset, compression):
#     if dataset == 'Original':
#         dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
#     elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
#         dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
#     else:
#         raise NotImplementedError
#     movies_path_list = sorted(glob(dataset_path + '*.mp4'))
#     print("{} : videos exist in {}".format(len(movies_path_list), dataset))
#     return movies_path_list

# def parse_labels(video_path):
#     return 0 if "original" in video_path else 1

# def parse_source_save_path(save_path):
#     if "original" in save_path:
#         return save_path
#     img_meta = save_path.split('/')
#     source_target_index = img_meta[-1]
#     source_index = source_target_index.split('_')[0]
#     manipulation_name = img_meta[-4]
#     original_name = "youtube"
#     source_save_path = save_path.replace(
#         "manipulated_sequences", "original_sequences"
#     ).replace(
#         manipulation_name, original_name
#     ).replace(
#         source_target_index, source_index
#     )
#     return source_save_path

# def preprocess_frame(frame, cnt_frame, save_path, source_save_path, label, face_detector, face_predictor):
#     frame_video_dict = dict()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = face_detector(frame, 1)

#     if len(faces) == 0:
#         tqdm.write(f'No faces in frame {cnt_frame}')
#         return

#     landmarks = []
#     size_list = []

#     for face in faces:
#         landmark = face_predictor(frame, face)
#         landmark = face_utils.shape_to_np(landmark)
#         x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
#         x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
#         face_s = (x1 - x0) * (y1 - y0)
#         size_list.append(face_s)
#         landmarks.append(landmark)

#     landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
#     landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

#     frame_video_dict['landmark'] = landmarks.tolist()
#     frame_video_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
#     frame_video_dict['label'] = label

#     IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = frame_video_dict

#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     image_path = f"{save_path}/frame_{cnt_frame}.png"
#     cv2.imwrite(image_path, frame)

# def preprocess_video(video_path, save_path, face_detector, face_predictor):
#     video_dict = dict()
#     label = parse_labels(video_path)
#     source_save_path = parse_source_save_path(save_path)
#     os.makedirs(save_path, exist_ok=True)

#     cap_video = cv2.VideoCapture(video_path)
#     frame_count_video = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_idxs = np.linspace(0, frame_count_video - 1, NUM_FRAMES, endpoint=True, dtype=int)

#     for cnt_frame in range(frame_count_video):
#         ret, frame = cap_video.read()
#         if not ret:
#             tqdm.write(f'Frame read {cnt_frame} Error! : {os.path.basename(video_path)}')
#             continue

#         if cnt_frame not in frame_idxs:
#             continue

#         preprocess_frame(frame, cnt_frame, save_path, source_save_path, label, face_detector, face_predictor)

#     cap_video.release()

# def process_camera(save_path, face_detector, face_predictor):
#     label = 0  # Assume live video input is "original"
#     os.makedirs(save_path, exist_ok=True)
#     cap = cv2.VideoCapture(1)
#     cnt_frame = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             tqdm.write(f'Error reading frame {cnt_frame} from camera.')
#             break

#         preprocess_frame(frame, cnt_frame, save_path, save_path, label, face_detector, face_predictor)
#         cnt_frame += 1

#         # Show live feed with landmarks (Optional)
#         cv2.imshow('Live Feed', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     parser = argparse.ArgumentParser(description="Frame extraction from videos or live camera input.")
#     parser.add_argument('--use-camera', action='store_true', help="Use live camera input instead of videos.")
#     args = parser.parse_args()

#     face_detector = dlib.get_frontal_face_detector()
#     face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

#     if args.use_camera:
#         save_path = f"{SAVE_IMGS_PATH}/camera_frames"
#         process_camera(save_path, face_detector, face_predictor)
#     else:
#         for dataset in DATASETS:
#             for comp in COMPRESSION:
#                 movies_path_list = parse_video_path(dataset, comp)
#                 for video_path in tqdm(movies_path_list):
#                     save_path_per_video = video_path.replace(
#                         VIDEO_PATH, SAVE_IMGS_PATH
#                     ).replace('.mp4', '').replace("/videos", "/frames")
#                     preprocess_video(video_path, save_path_per_video, face_detector, face_predictor)

#     with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
#         json.dump(IMG_META_DICT, f)

# if __name__ == '__main__':
#     main()













# #!/usr/bin/env python3
# from glob import glob
# import os
# import cv2
# from tqdm import tqdm
# import numpy as np
# import dlib
# import json
# from imutils import face_utils

# VIDEO_PATH = "./data/FaceForensics++"
# SAVE_IMGS_PATH = "./test_images"
# PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
# DATASETS = {'Original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'Deepfakes', 'NeuralTextures'}
# COMPRESSION = {'raw'}
# IMG_META_DICT = dict()

# def parse_image_path(dataset, compression):
#     # this path setting follows FF++ dataset
#     if dataset == 'Original':
#         dataset_path = f'{VIDEO_PATH}/original_sequences/youtube/{compression}/videos/'
#     elif dataset in ['FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
#         dataset_path = f'{VIDEO_PATH}/manipulated_sequences/{dataset}/{compression}/videos/'
#     else:
#         raise NotImplementedError

#     # get all images under the specific manipulated/original sequences
#     images_path_list = sorted(
#         glob(dataset_path + '*.png') +
#         glob(dataset_path + '*.jpg') +
#         glob(dataset_path + '*.jpeg')
#     )
#     print("{} images exist in {}".format(len(images_path_list), dataset))
#     return images_path_list

# def parse_labels(image_path):
#     label = None
#     if "original" in image_path:
#         label = 0
#     else:
#         label = 1
#     return label

# def parse_source_save_path(save_path):
#     source_save_path = None
#     if "original" in save_path:
#         source_save_path = save_path
#     else:
#         img_meta = save_path.split('/')
#         source_target_index = img_meta[-1]
#         source_index = source_target_index.split('_')[0]
#         manipulation_name = img_meta[-4]
#         original_name = "youtube"
#         source_save_path = save_path.replace(
#             "manipulated_sequences", "original_sequences"
#         ).replace(
#             manipulation_name, original_name
#         ).replace(
#             source_target_index, source_index
#         )
#     return source_save_path

# def preprocess_image(image_path, save_path, face_detector, face_predictor):
#     # save the image meta info here
#     image_dict = dict()
#     # get the labels
#     label = parse_labels(image_path)
#     # get the path of corresponding source imgs
#     source_save_path = parse_source_save_path(save_path)
#     # prepare the save path
#     os.makedirs(save_path, exist_ok=True)

#     # process a single image
#     frame = cv2.imread(image_path)
#     process_frame(frame, 0, save_path, source_save_path, label, face_detector, face_predictor)
#     return

# def process_frame(frame, cnt_frame, save_path, source_save_path, label, face_detector, face_predictor):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = face_detector(frame, 1)

#     if len(faces) == 0:
#         tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(save_path)))
#         return

#     # Process landmarks for the current frame
#     frame_image_dict = dict()

#     landmarks = list()  # save the landmark
#     size_list = list()  # save the size of the detected face

#     for face_idx in range(len(faces)):
#         landmark = face_predictor(frame, faces[face_idx])
#         landmark = face_utils.shape_to_np(landmark)
#         x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
#         x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
#         face_s = (x1 - x0) * (y1 - y0)
#         size_list.append(face_s)
#         landmarks.append(landmark)

#     # save the landmark with the biggest face
#     landmarks = np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
#     landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]

#     # save the meta info of the current frame
#     frame_image_dict['landmark'] = landmarks.tolist()
#     frame_image_dict['source_path'] = f"{source_save_path}/frame_{cnt_frame}"
#     frame_image_dict['label'] = label

#     # Store frame-specific metadata
#     IMG_META_DICT[f"{save_path}/frame_{cnt_frame}"] = frame_image_dict

#     # save the frame
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     image_path = f"{save_path}/frame_{cnt_frame}.png"
#     cv2.imwrite(image_path, frame)

# def main():
#     face_detector = dlib.get_frontal_face_detector()
#     face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
#     for dataset in DATASETS:
#         for comp in COMPRESSION:
#             images_path_list = parse_image_path(dataset, comp)
#             n_sample = len(images_path_list)
#             for i in tqdm(range(n_sample)):
#                 save_path_per_image = images_path_list[i].replace(
#                     VIDEO_PATH, SAVE_IMGS_PATH
#                 ).replace('.png', '').replace("/videos", "/frames")
#                 preprocess_image(
#                     images_path_list[i], save_path_per_image,
#                     face_detector, face_predictor
#                 )
#     with open(f"{SAVE_IMGS_PATH}/ldm.json", 'w') as f:
#         json.dump(IMG_META_DICT, f)

# if __name__ == '__main__':
#     main()
