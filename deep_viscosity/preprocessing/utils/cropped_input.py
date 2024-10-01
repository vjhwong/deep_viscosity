import cv2
import os


def crop_videos(input_path: str):
    """Crop all frames in all videos in the folder and save them in a new folder.

    Args:
        video_folder_path (str): Path to the folder containing the videos.
    """

    output_folder_path = os.path.join("data", "cropped")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate over all video files in the folder
    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        crop_frames(video_path, output_folder_path)


def crop_frames(video_path: str, output_folder_path: str):
    """Mask all frames in the video with the mask provided and save them in a new folder.

    Args:
        video_path (str): Path to the video file.
        output_folder_path (str): Path to the folder where the masked frames will be saved.
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
    masked_frames_folder_path = os.path.join(output_folder_path, video_name)
    os.makedirs(masked_frames_folder_path)

    for index, frame in enumerate(frames):
        if index in range(45, 99):
            # Crop
            cropped_frame = frame[260:470, 300:520]
            output_path = os.path.join(
                masked_frames_folder_path, f"cropped_frame_{index}.jpg"
            )
            cv2.imwrite(output_path, cropped_frame)


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
    crop_videos(input_path)


if __name__ == "__main__":
    main()
