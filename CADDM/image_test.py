import os
import cv2
import dlib
import json
from imutils import face_utils
import argparse

# Paths
PREDICTOR_PATH = "./lib/shape_predictor_81_face_landmarks.dat"
OUTPUT_JSON_PATH = "./test_images/ldm.json"
OUTPUT_IMAGE_PATH = "./test_images/landmarks_image.jpg"  # Path for the output image

# Initialize the face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor(PREDICTOR_PATH)

def process_image(image_path):
    """
    Process an image to extract facial landmarks and save a visualized image.

    Args:
        image_path (str): Path to the image.

    Returns:
        dict: A dictionary containing landmarks and metadata.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = face_detector(image_rgb, 1)
    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return {}

    # Process landmarks for the largest detected face
    landmarks_dict = {}
    size_list = []  # Sizes of detected faces
    landmarks_list = []

    for face in faces:
        landmark = face_predictor(image_rgb, face)
        landmark = face_utils.shape_to_np(landmark)
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        face_size = (x1 - x0) * (y1 - y0)
        size_list.append(face_size)
        landmarks_list.append((landmark, (x0, y0, x1, y1)))

    # Select the landmarks for the largest face
    largest_face_idx = size_list.index(max(size_list))
    largest_landmarks, bbox = landmarks_list[largest_face_idx]
    x0, y0, x1, y1 = bbox

    # Draw bounding box and landmarks on the image
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)  # Draw bounding box
    for (x, y) in largest_landmarks:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Draw landmarks

    # Save the visualized image
    cv2.imwrite(OUTPUT_IMAGE_PATH, image)
    print(f"Image with landmarks saved to {OUTPUT_IMAGE_PATH}")

    # Save landmarks data
    landmarks_dict["landmarks"] = largest_landmarks.tolist()
    landmarks_dict["image_path"] = image_path

    return landmarks_dict

def save_landmarks_to_json(landmarks_dict, output_path):
    """
    Save the landmarks dictionary to a JSON file.

    Args:
        landmarks_dict (dict): The dictionary containing landmarks data.
        output_path (str): Path to save the JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(landmarks_dict, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Process an image and extract facial landmarks.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    image_path = args.image_path

    # Process the image and extract landmarks
    landmarks = process_image(image_path)

    if landmarks:
        # Save landmarks to a JSON file
        save_landmarks_to_json(landmarks, OUTPUT_JSON_PATH)
        print(f"Landmarks saved to {OUTPUT_JSON_PATH}")
    else:
        print("No landmarks detected.")

if __name__ == '__main__':
    main()
