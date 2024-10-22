import os
import shutil

import cv2
import numpy as np
from scipy import interpolate


def get_video_index(video_name: str) -> int:
    """Returns the index of a video

    Args:
        video_name(str): The file name of the video

    Returns:
        int: The index of the video
    """
    return int(video_name.split("_")[1].split(".")[0])


def get_new_video_name(remove_first_vid: bool, video_name: str) -> str:
    """Returns a new name for a video where the index in the file name has been decreased by 1,
    if the first video has been removed.

    Args:
        remove_first_vid (bool): Indicates if the first video has been removed.
        video_name (str): The name of the video.

    Returns:
        str: The new video name.
    """
    if remove_first_vid:
        video_index = get_video_index(video_name) - 1
    else:
        video_index = get_video_index(video_name)

    return f"{video_name.split('_')[0]}__{video_index}.avi"


def remove_first_video(remove_first_vid: bool, input_path: str, output_path: str) -> None:
    """Removes the first video of each viscosity.

    Args:
        remove_first_vid (bool): Indicates if the first video has been removed.
        input_path (str): The path to the videos.
        output_path (str): The new path to the remaining videos.
    """
    if not os.path.exists(output_path):
        shutil.copytree(input_path, output_path)
    else:
        print(f"The {output_path} already exists!")
        return

    for video_name in sorted(os.listdir(output_path), reverse=False):
        video_path = os.path.join(output_path, video_name)
        video_index = get_video_index(video_name)

        if video_index == 1 and remove_first_vid:
            os.remove(video_path)
        else:
            new_video_name = get_new_video_name(remove_first_vid, video_name)
            new_video_path = os.path.join(output_path, new_video_name)
            os.rename(video_path, new_video_path)


def find_interpolated_viscosities(desired_percentages: list[float]) -> tuple[list[int]]:
    """Finds the viscosity of glycerol solutions of different percentages based on known viscosities

    Args:
        desired_percentages (list[float]): A list with the weight percentages of glycerol for which one seeks the corresponding viscosity

    Returns:
        tuple[list[int]]: A tuple with a list of percentage values and a list of corresponding viscosities
    """
    known_percentages = np.load(os.path.join(
        "vectors", "known_percentages.npy"
    ))
    known_viscosities = np.load(os.path.join(
        "vectors", "known_viscosities.npy"
    ))

    desired_percentages = list(map(lambda x: x * 100, desired_percentages))

    x = np.array(known_percentages)
    y = np.array(known_viscosities)

    f = interpolate.interp1d(x, y, kind="cubic")
    interpolated_viscosities = f(desired_percentages)

    desired_percentages.sort()
    interpolated_viscosities.sort()

    return (desired_percentages, interpolated_viscosities)


def rename_videos(data_path: str, percentages: list[str]) -> None:
    """Renames video files so they contain the viscosity instead of the percentage glycerol.

    Args:
        data_path (str): Path to video folders.
        percentages (list[str]): A list with the weight percentages of glycerol for all videos
    """
    (percentages, viscosities) = find_interpolated_viscosities(percentages)
    old_name_to_new_name = {}
    for percent, viscosity in zip(percentages, viscosities):
        old_name_to_new_name[str(percent).rstrip(
            '0').rstrip('.')] = str(round(viscosity, 2))

    for file in os.listdir(data_path):
        if "avi" in file:
            old_label = file.split("_", 1)
            if '.' in old_label[0]:
                old_label[0] = old_label[0].rstrip('0').rstrip('.')
            old_file_name = os.path.join(data_path, file)
            new_file_name = os.path.join(
                data_path, old_name_to_new_name[old_label[0]] + old_label[1]
            )

            os.rename(old_file_name, new_file_name)


def get_frames(video_path: str) -> list[np.ndarray]:
    """Extracts all frames in the video to a list

    Args: 
        video_path (str): Path to the video file.

    Returns:
        list[np.ndarray]: List of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

    cap.release()
    return frames


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


def check_index_in_range(index: int, frame_range: tuple[int]) -> bool:
    """Checks if the index of is within the selected frame range
    Returns True if index is within range, else False

    Args: 
        index (int): Index 
        frame_range tuple[int]: The selected frame range
    """
    first_frame, last_frame = frame_range
    return index >= first_frame and index <= last_frame


def mask_frames(video_path: str, masks: np.ndarray, output_path: str, dimensions: tuple[int], frame_range: tuple[int]) -> None:
    """Mask all frames in the video with the mask provided and save them in a new folder.

    Args:
        video_path (str): Path to the video file.
        masks (np.ndarray): Masks to apply to the frames.
        output_path (str): Path to the folder where the masked frames will be saved.
        dimensions (tuple[int]): The cropping dimensions
        frame_range (tuple[int]): The selected frame range
    """
    frames = get_frames(video_path)
    video_name = get_video_name(video_path)
    masked_frames_folder_path = os.path.join(output_path, video_name)
    os.makedirs(masked_frames_folder_path)

    for index, (mask, frame) in enumerate(zip(masks, frames)):
        if not check_index_in_range(index, frame_range):
            continue
        masked_frame = cv2.bitwise_and(
            frame, frame, mask=mask.astype(np.uint8))
        top, left, bottom, right = dimensions
        masked_frame = masked_frame[top:bottom, left:right]
        output_path = os.path.join(
            masked_frames_folder_path, f"masked_frame_{index}.jpg"
        )
        cv2.imwrite(output_path, masked_frame)


def get_border_pixels(frame: np.ndarray) -> list[int, int, int, int]:
    borders = []
    for i, row in enumerate(frame):
        if row.sum() > 0:
            borders.append(i)
            break
    for i, col in enumerate(frame.T):
        if col.sum() > 0:
            borders.append(i)
            break
    for i, row in enumerate(frame[::-1]):
        if row.sum() > 0:
            borders.append(len(frame) - i)
            break
    for i, col in enumerate(frame.T[::-1]):
        if col.sum() > 0:
            borders.append(len(frame.T) - i)
            break
    return borders


def get_window_size(frames: np.ndarray, padding=10) -> tuple[int, int, int, int]:
    top_border, left_border, bottom_border, right_border = (
        np.inf,
        np.inf,
        -np.inf,
        -np.inf,
    )

    for frame in frames:
        top, left, bottom, right = get_border_pixels(frame)
        if top_border > top:
            top_border = top
        if left_border > left:
            left_border = left
        if bottom_border < bottom:
            bottom_border = bottom
        if right_border < right:
            right_border = right
    return (
        top_border - padding,
        left_border - padding,
        bottom_border + padding,
        right_border + padding,
    )


def mask_videos(input_path: str, output_path: str, masks_path: str, frame_range: tuple[int]) -> None:
    """Mask all frames in all videos in the folder with the mask provided and save them in a new folder.

    Args:
        input_path (str): Path to the folder containing the videos.
        output_path (str): Path to the folder where the masked frames will be saved.
        masks_path (str): Path to the mask file.
        frame_range (tuple[int]): The selected frame range
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    masks = np.load(masks_path)

    dimensions = get_window_size(masks)
    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        mask_frames(video_path, masks, output_path, dimensions, frame_range)


def get_new_folder_path(raw_data_path: str, new_folder: str) -> str:
    return "/".join(raw_data_path.split("/")[:-1]) + "/" + new_folder
