import os

import PIL
import modeling.deep_viscosity_model as dv_model
import modeling.train as train
from dataset import Dataset
from utils.video_to_frames import VideoToFrames


def main():
    video_folder = os.path.join("data", "raw")
    processed_folder = os.path.join("data", "processed")

    n_frames = 20
    process_videos = False
    vtf = VideoToFrames(video_folder, processed_folder, n_frames)
    if process_videos:
        vtf.process_videos_in_directory()

    # TODO: refactor into separate functio in utils.functions
    first_image_folder = os.listdir(processed_folder)[0]
    first_frame_name = os.listdir(os.path.join(processed_folder, first_image_folder))[0]
    first_frame_path = os.path.join(
        processed_folder, first_image_folder, first_frame_name
    )
    width, height = PIL.Image.open(first_frame_path).size

    batch_size = 4
    test_size = 0.2
    validation_size = 0.2

    model = dv_model.CNN3DVisco(
        t_dim=n_frames,
        img_x=width,
        img_y=height,
    )
    dataset = Dataset(processed_folder, batch_size, test_size, validation_size)
    train_loader, val_loader, test_loader = dataset.create_dataloaders()
    train.train(model, train_loader)


if __name__ == "__main__":
    main()
