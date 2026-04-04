import os
import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from a video at a specified FPS.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        fps (int): Target frames per second for extraction.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    # Get original video FPS and total frame count
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval (how many frames to skip)
    # If original FPS is 30 and target is 2, we save every 15th frame
    frame_interval = int(original_fps / fps)

    # Prevent division by zero or invalid interval
    if frame_interval < 1:
        frame_interval = 1

    print(f"Processing: {video_path}")
    print(f"Original FPS: {original_fps}, Total Frames: {total_frames}")
    print(f"Target FPS: {fps}, Frame Interval: {frame_interval}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame based on interval
        if frame_count % frame_interval == 0:
            # Generate filename with 6-digit zero-padded number
            filename = f"{saved_count:06d}.png"
            output_path = os.path.join(output_dir, filename)

            # Save the frame as PNG
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # Release video capture object
    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}\n")


def main():
    # Define input and output paths
    videos_dir = Path("../database/videos")
    frames_dir = Path("../database/frames")

    # Check if video directory exists
    if not videos_dir.exists():
        print(f"Error: Video directory {videos_dir} does not exist.")
        return

    # Get all .mp4 files in the directory
    video_files = list(videos_dir.glob("*.mp4"))

    if not video_files:
        print(f"No .mp4 video files found in {videos_dir}")
        return

    print(f"Found {len(video_files)} video file(s)\n")

    # Process each video file
    for video_path in video_files:
        # Get video filename without extension
        video_name = video_path.stem

        # Construct output directory path: ./frames/<video_name>/
        output_dir = frames_dir / video_name

        # Extract frames
        extract_frames(str(video_path), str(output_dir), fps=2)


if __name__ == "__main__":
    main()