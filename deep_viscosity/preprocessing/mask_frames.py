import cv2
import os
import numpy as np


def mask_videos(video_folder_path: str, mask_path: str):
    output_folder_path = os.path.join("data", "masked")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load the mask
    masks = np.load(mask_path)

    # Iterate over all video files in the folder
    for video_file in os.listdir(video_folder_path):
        if "TILT_GREEN" in video_file:
            video_path = os.path.join(video_folder_path, video_file)
            mask_frames(video_path, masks, output_folder_path)


def mask_frames(video_path: str, masks: np.ndarray, output_folder_path: str):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    video_name = os.path.basename(video_path).split(".")[0]
    masked_frames_folder_path = os.path.join(output_folder_path, video_name)
    os.makedirs(masked_frames_folder_path)

    for index, (mask, frame) in enumerate(zip(masks, frames)):
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))

        # Crop
        masked_frame = masked_frame[260:470, 300:520]
        output_path = os.path.join(
            masked_frames_folder_path, f"masked_frame_{index}.jpg"
        )
        cv2.imwrite(output_path, masked_frame)


def main():
    video_folder_path = os.path.join("data", "raw")
    mask_path = os.path.join("masks.npy")

    mask_videos(video_folder_path, mask_path)


if __name__ == "__main__":
    main()
