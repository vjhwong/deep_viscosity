import os
import numpy as np

from utils.video_to_frames import VideoToFrames
from dataset import Dataset


def main():
    processed_data_path = os.path.join("data", "processed")
    tensor_data_path = os.path.join(processed_data_path, "tensor")

    dataset = Dataset(tensor_data_path, 32)
    dataset.create_dataloaders()


if __name__ == "__main__":
    main()
