# import cv2
# import numpy as np

# def draw_landmarks_and_bbox(image, landmarks):
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
#     for (x, y) in landmarks:
#         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green circle for landmarks

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

# # Example usage:
# if __name__ == "_main_":
#     # Load an example image
#     image = cv2.imread('example_image.jpg')

#     # Example landmarks (these would typically come from a landmark detection model)
#     # Example: [(x1, y1), (x2, y2), ..., (xn, yn)]
#     landmarks = [(100, 100), (120, 110), (130, 130), (110, 150), (95, 140)]

#     # Draw landmarks and bounding box
#     output_image = draw_landmarks_and_bbox(image, landmarks)

#     # Convert to BGR before displaying with OpenCV
#     output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

#     # Show the output image
#     cv2.imshow('Landmarks with Bounding Box', output_image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()