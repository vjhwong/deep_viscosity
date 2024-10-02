import os


def get_extreme_number_of_frames(path: str) -> tuple[int]:
    """Counts the maximum and minimum number of frames in the videos

    Args:
        path (str): Path where the videos are stored as frames

    Returns:
      tuple[int]: the maximum and minimum number of frames

    """
    counts = []
    for folder in os.listdir(path):
        counter = 0
        for _ in os.listdir(os.path.join(path, folder)):
            counter += 1
        counts.append(counter)

    return (max(counts), min(counts))


def get_videos_with_num_frames(n_frames: int, path: str) -> list[str]:
    """Gets the file names of the videos with the specified number of frames

    Args:
        path (str): Path where the videos are stored as frames

    Returns:
      list[str]: A list with the names of the videos with the specified number of frames
    """
    video_names = []
    for folder in os.listdir(path):
        counter = 0
        for frame in os.listdir(os.path.join(path, folder)):
            counter += 1
        if counter == n_frames:
            video_names.append(folder)
    return video_names


def main():
    path = "data/all_frames"
    (max_frames, min_frames) = get_extreme_number_of_frames(path)
    longest_videos = get_videos_with_num_frames(max_frames, path)
    print(longest_videos)
    print(max_frames, min_frames)


if __name__ == "__main__":
    main()
