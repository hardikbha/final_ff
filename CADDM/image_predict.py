#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate, get_video_auc


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def draw_landmarks_and_bbox_batch(images, landmarks_batch, predicted_labels):
    """
    Draw landmarks on each image in the batch, calculate a bounding box around the landmarks,
    save the frames in an organized folder structure, and create an MP4 video.

    Args:
    - images: The input batch of images (tensor of shape [batch_size, height, width, channels]).
    - landmarks_batch: A list of lists, where each inner list contains tuples of landmarks for one image.
                        Example: [[(x1, y1), (x2, y2)], [(x1, y1), (x2, y2)], ...]
    - predicted_labels: A list or tensor containing label scores for the images.

    Returns:
    - The batch of images with landmarks and bounding boxes drawn.
    """
    batch_size, height, width, channels = images.shape

    # Create an empty list to store the images with bounding boxes
    output_images = []

    # Create directories for saving frames and video
    os.makedirs("output/frames", exist_ok=True)

    # Process each image in the batch
    for i in range(batch_size):
        print("b_s", batch_size)
        image = images[i]  # Get image i from the batch
        print("I_SH", image.shape)

        image = image.detach().cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Scale to [0, 1]
        image = (image * 255).astype(np.uint8)  # Convert to [0, 255] and ensure uint8
        print("im_shape", image.shape)

        labelsS = predicted_labels[i] if isinstance(predicted_labels, (list, torch.Tensor)) else predicted_labels

        print("P_Labels", labelsS)
        print("dt_labels", type(labelsS))

        # Get the landmarks for this image
        landmarks = landmarks_batch[i]
        print("landmarks", landmarks)

        # # Draw landmarks
        # for (x, y) in landmarks:
        #     x, y = int(x), int(y)
        #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green circle for landmarks

        # Calculate the bounding box (min and max x and y coordinates)
        xs = [int(x) for x, y in landmarks]
        ys = [int(y) for x, y in landmarks]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        # Create a consistent bounding box for the entire video
        if (labelsS > 0.07):
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box (Fake)
            label_text1 = "Fake"
            text_position = (x_min, y_min - 10)  # Place text slightly above the top-left corner of the box
            cv2.putText(
                image,
                label_text1,
                text_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),  # Red text
                thickness=1,
                lineType=cv2.LINE_AA
            )
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box (Real)
            label_text2 = "Real"
            text_position = (x_min, y_min - 10)  # Place text slightly above the top-left corner of the box
            cv2.putText(
                image,
                label_text2,
                text_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),  # Green text
                thickness=1,
                lineType=cv2.LINE_AA
            )

        # Save each frame as an individual image in the frames folder
        frame_filename = os.path.join("output/frames", f'frame_{i:04d}.png')
        cv2.imwrite(frame_filename, image)

        # Add frame to video writer
        video_writer.write(image)

        # Append to the output_images list
        output_images.append(image)

    # Release the video writer
    video_writer.release()

    return np.array(output_images)


def test():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    # get testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=False, num_workers=0,
                             )

    # start testing.
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    for batch_data, batch_labels, img_frame, ld in test_loader:

        labels, video_name = batch_labels
        labels = labels.long()

        outputs = net(batch_data)
        outputs = outputs[:, 1]
        outputs = outputs.detach().cpu().numpy().tolist()
        print("ld_new", ld)

        output_image = draw_landmarks_and_bbox_batch(img_frame, ld, outputs)



if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab

