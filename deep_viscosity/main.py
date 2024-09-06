import numpy as np

from utils.video_to_frames import VideoToFrames
from dataset import Dataset


def main():
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    video_to_frames = VideoToFrames(raw_data_path, processed_data_path)
    video_to_frames.process_videos_in_directory()

    begin_frame, end_frame, skip_frame = 0, 60, 10
    selected_frames = np.arange(begin_frame, end_frame, skip_frame)

    dataset = Dataset(processed_data_path, 1)
    dataset.create_dataloaders()


if __name__ == "__main__":
    main()
