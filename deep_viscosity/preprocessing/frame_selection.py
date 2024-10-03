import os
import shutil


def frame_selection(output_path: str, input_path: str, frame_range: tuple[int]):
    """Creates a new folder with only selected frames.

    Args:
        output_path (str): Where the folders with selected frames will be stores.
        input_path (str): The path with all frames for each video.
        frame_range (tuple[int]): The range for which frames should be selected.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for video in os.listdir(input_path):
        os.makedirs(os.path.join(output_path, video))
        for frame in os.listdir(os.path.join(input_path, video)):
            index = int(frame.split("_")[2].split(".")[0])
            if index >= frame_range[0] and index <= frame_range[1]:
                shutil.copy(
                    os.path.join(input_path, video, frame),
                    os.path.join(output_path, video, frame),
                )


def main():
    output_path = "data/processed"
    input_path = "data/masked"
    frame_range = (45, 99)
    frame_selection(output_path, input_path, frame_range)


if __name__ == "__main__":
    main()
