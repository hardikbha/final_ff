import cv2
import numpy as np

def merge_videos(video1_path, video2_path, output_video_path):
    # Open the first video
    cap1 = cv2.VideoCapture(video1_path)
    
    # Open the second video
    cap2 = cv2.VideoCapture(video2_path)

    # Get properties from the first video
    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)

    # Get properties from the second video
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # Ensure compatibility between the two videos
    if frame_height1 != frame_height2 or fps1 != fps2:
        raise ValueError("Videos have different heights or frame rates and cannot be merged.")

    # Define the codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    output_width = frame_width1 + frame_width2  # Combined width of the two videos
    out = cv2.VideoWriter(output_video_path, fourcc, fps1, (output_width, frame_height1))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop if either video ends
        if not ret1 or not ret2:
            break

        # Concatenate frames horizontally
        merged_frame = np.hstack((frame1, frame2))

        # Write the merged frame to the output video
        out.write(merged_frame)

    # Release all resources
    cap1.release()
    cap2.release()
    out.release()
    print(f"Merged video saved as {output_video_path}")

# Example Usage:
if __name__ == "__main__":
    video1 = "/Users/hardiksharma/Downloads/WhatsApp Video 2024-12-07 at 11.35.58.mp4"  # Path to the first input video
    video2 = "/Users/hardiksharma/Downloads/WhatsApp Video 2024-12-07 at 11.38.48.mp4"  # Path to the second input video
    output_video = "reconstructed_video.mp4"  # Path to save the merged video

    merge_videos(video1, video2, output_video)