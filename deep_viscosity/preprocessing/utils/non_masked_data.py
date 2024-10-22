import os

import cv2
import numpy as np

import functions as f


def create_non_masked_data(input_path: str, output_path: str, frame_range: tuple[int], masks_path: str) -> None:
    """Create a folder with the non-masked data.

    Args:
        input_path (str): Path to the folder containing the videos.
        output_path (str): Path to the folder where the non-masked data will be saved.
        frame_range (tuple[int]): The selected frame range
        masks_path (str): Path to the masks file.
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    masks = np.load(masks_path)

    top, left, bottom, right = f.get_window_size(masks)
    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        frames = f.get_frames(video_path)
        video_name = f.get_video_name(video_path)
        non_masked_folder_path = os.path.join(output_path, video_name)
        os.makedirs(non_masked_folder_path)

        for index, frame in enumerate(frames):
            if not f.check_index_in_range(index, frame_range):
                continue
            cv2.imwrite(
                os.path.join(
                    non_masked_folder_path, f"non_masked_frame_{index}.jpg"
                ),
                frame[top:bottom, left:right]
            )


def main() -> None:
    input_path = "data/raw_modified"
    output_path = "data/non_masked"
    frame_range = (45, 99)
    masks_path = "data/masks.npy"

    create_non_masked_data(input_path, output_path, frame_range, masks_path)


if __name__ == "__main__":
    main()
