import cv2
import numpy as np

def split_concatenated_video(input_video_path, output_video1_path, output_video2_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the video properties (frame width, height, frame rate)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # The width of each individual video is half the width of the concatenated video
    split_width = frame_width // 2
    
    # Define the codec and create VideoWriter objects for the two output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out1 = cv2.VideoWriter(output_video1_path, fourcc, fps, (split_width, frame_height))
    out2 = cv2.VideoWriter(output_video2_path, fourcc, fps, (split_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Split the frame into two parts (left and right)
        frame1 = frame[:, :split_width]  # Left side of the concatenated video
        frame2 = frame[:, split_width:]  # Right side of the concatenated video
        
        # Write the split frames to the respective output videos
        out1.write(frame1)
        out2.write(frame2)
    
    # Release the video capture and writer objects
    cap.release()
    out1.release()
    out2.release()
    print(f"Videos saved as {output_video1_path} and {output_video2_path}")


# Example Usage:
if __name__ == "__main__":
    input_video = "/Users/hardiksharma/Downloads/WhatsApp Video 2024-12-04 at 14.11.49 (4).mp4"  # Path to the input concatenated video
    output_video1 = "video_part7.mp4"      # Path to save the first part
    output_video2 = "video_part8.mp4"      # Path to save the second part

    split_concatenated_video(input_video, output_video1, output_video2)