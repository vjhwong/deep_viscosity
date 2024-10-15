import cv2
import os
import numpy as np

from utils.cropping import get_window_size

def mask_videos(input_path: str, output_path: str, mask_path: str):
    """Mask all frames in all videos in the folder with the mask provided and save them in a new folder.

    Args:
        input_path (str): Path to the folder containing the videos.
        output_path (str): Path to the folder where the masked frames will be saved.
        mask_path (str): Path to the mask file.
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the mask
    masks = np.load(mask_path)

    # Iterate over all video files in the folder
    dimensions = get_window_size(masks)
    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        mask_frames(video_path, masks, output_path, dimensions)


def mask_frames(video_path: str, masks: np.ndarray, output_path: str, dimensions: tuple[int]):
    """Mask all frames in the video with the mask provided and save them in a new folder.

    Args:
        video_path (str): Path to the video file.
        masks (np.ndarray): Masks to apply to the frames.
        output_path (str): Path to the folder where the masked frames will be saved.
        dimensions tuple[int]: The cropping dimensions
    """
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    video_name = get_video_name(video_path)
    masked_frames_folder_path = os.path.join(output_path, video_name)
    os.makedirs(masked_frames_folder_path)

    for index, (mask, frame) in enumerate(zip(masks, frames)):
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))

        # Crop
        masked_frame = masked_frame[dimensions[0]:dimensions[2], dimensions[1]:dimensions[3]]
        output_path = os.path.join(
            masked_frames_folder_path, f"masked_frame_{index}.jpg"
        )
        cv2.imwrite(output_path, masked_frame)


def get_video_name(video_path: str) -> str:
    """Get the name of the video from the video path.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: Name of the video.
    """
    split_path = os.path.basename(video_path).split(".")
    if len(split_path) == 3:
        return split_path[0] + "." + split_path[1]
    return split_path[0]


def main():
    input_path = "data/raw_modified"
    output_path = "data/masked"
    mask_path = "data/masks.npy"
    mask_videos(input_path, output_path, mask_path)

if __name__ == "__main__":
    main()