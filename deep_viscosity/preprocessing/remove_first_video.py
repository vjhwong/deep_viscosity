import os
import shutil


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
    if the first video has been removed

    Args:
        remove_first_vid (bool): Indicates if the first video has been removed
        video_name (str): The name of the video

    Returns:
        str: The new video name
    """
    if remove_first_vid:
        video_index = get_video_index(video_name) - 1
    else:
        video_index = get_video_index(video_name)

    return f"{video_name.split('_')[0]}__{video_index}.avi"


def remove_first_video(remove_first_vid: bool, input_path: str, output_path: str):
    """Removes the first video of each viscosity

    Args:
        remove_first_vid (bool): Indicates if the first video has been removed
        input_path (str): The path to the videos
        output_path (str): The new path to the remaining videos
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


def main():
    input_path = "data/raw"
    output_path = "data/raw_modified"
    remove_first_video(True, input_path, output_path)


if __name__ == "__main__":
    main()
