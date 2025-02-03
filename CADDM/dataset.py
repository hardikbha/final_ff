#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from natsort import natsorted

from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input


class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        self.root = self.config['dataset']['img_path']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()
        self.samples = self.collect_samples()

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            print("d", d)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                print("r", r)
                print("f_name", filename)
                for name in natsorted(filename):
                    print("name", name)
                    if name.startswith('.'):  # Skip hidden/system files
                        continue
                    path = os.path.join(r, name)
                    print("path", path)
                    info_key = path[:-4]  # Remove file extension for the key
                    video_name = '/'.join(path.split('/')[:-1])
                    print("V_name", video_name)

                    # Check if the key exists in the metadata dictionary
                    if info_key not in self.info_meta_dict:
                        print(f"Warning: {info_key} not found in info_meta_dict. Skipping...")
                        continue

                    # Access metadata and construct the sample
                    info_meta = self.info_meta_dict[info_key]
                    landmark = info_meta['landmark']
                    class_label = int(info_meta['label'])
                    source_path = info_meta['source_path'] + path[-4:]
                    samples.append(
                        (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                    )

        return samples


    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        # classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}
    # def draw_landmarks_and_bbox(self,image, landmarks):
    #     """
    #     Draw landmarks on an image and calculate a bounding box around the landmarks.

    #     Args:
    #     - image: The input image (numpy array).
    #     - landmarks: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the 2D coordinates of landmarks.

    #     Returns:
    #     - The image with landmarks and bounding box drawn.
    #     """

    #     # Convert the image to RGB for visualization purposes (if using OpenCV)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Draw landmarks
    #     # for (x, y) in landmarks:
    #     #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green circle for landmarks

    #     # Calculate the bounding box (min and max x and y coordinates)
    #     xs = [x for x, y in landmarks]
    #     ys = [y for x, y in landmarks]

    #     x_min = min(xs)
    #     y_min = min(ys)
    #     x_max = max(xs)
    #     y_max = max(ys)

    #     # Draw the bounding box around the landmarks
    #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

    #     return image

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # output_path = 'path_to_save_image.jpg'
        # cv2.imwrite(output_path, img)
        # print("image_shape1", img.shape)

        img_frame = img
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)


        if self.mode == "train":
            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = torch.Tensor(img.transpose(2, 0, 1))
            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input([img], ld, label, self.config)
            # print("img_shape", img[0].shape)
            # print("img_len", len(img))
            print(f"Accessing index: {index}")

            img1 = img[0]
            # print("img_shape_2", im)

            # Save the image using OpenCV in PNG format
            output_path = 'save_image.png'
            cv2.imwrite(output_path, img1)


            # Convert tensor (C, H, W) to numpy array (H, W, C)

            img = torch.Tensor(img[0].transpose(2, 0, 1))

            # Retrieve video name
            video_name = label_meta['video_name']
            # print("label", label)
            # print("Video_name", video_name)
            # print("ld", ld)

            # print("ld_shape", len(ld))
            # output_image = self.draw_landmarks_and_bbox(img_frame, ld)

            # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            # print("outpS", output_image.shape)

            # output_path3 = 'Image_bounding_box.png'
            print("IMf", img_frame.shape)
        
            # cv2.imwrite(output_path3, output_image)
            return img, (label, video_name), img_frame, ld        

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    from lib.util import load_config
    config = load_config('./configs/caddm_train.cfg')
    d = DeepfakeDataset(mode="test", config=config)
    for index in range(len(d)):
        res = d[index]
# vim: ts=4 sw=4 sts=4 expandtab
