import cv2
import os


def video_to_frames(video_path, output_folder, frames_per_second):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video from the file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Calculate the interval between frames to capture
    frame_interval = fps / frames_per_second

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration (s): {duration}")
    print(f"Frame Interval: {frame_interval}")

    frame_number = 0
    extracted_frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        # Capture the frame if it matches the interval
        if frame_number % int(frame_interval) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_count += 1

        frame_number += 1

        # Stop if the frame interval exceeds the video duration
        if (frame_number / fps) > duration:
            break

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {extracted_frame_count} frames to '{output_folder}'.")


# Example usage
video_path = 'path_to_your_video.mp4'
output_folder = 'output_frames'
frames_per_second = 2  # Number of frames to extract per second

video_to_frames(video_path, output_folder, frames_per_second)
