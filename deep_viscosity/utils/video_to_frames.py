import cv2
import os
import torch
from torchvision import transforms


class VideoToFrames:
    def __init__(self, data_path: str, output_folder: str) -> None:
        self._data_path = data_path
        self._output_folder = output_folder
        self._jpg_folder = os.path.join(self._output_folder, "jpg")
        self._tensor_folder = os.path.join(self._output_folder, "tensor")
        self._create_folder(self._jpg_folder)
        self._create_folder(self._tensor_folder)

    ### Public methods ###

    def process_videos_in_directory(self) -> None:
        video_files = [
            file for file in os.listdir(self._data_path) if file.endswith(".avi")
        ]
        for video_file in video_files:
            video_path = os.path.join(self._data_path, video_file)
            self._video_to_frames(video_path)

    ### Private methods ###

    def _create_folder(self, _output_folder: str) -> None:
        if not os.path.exists(_output_folder):
            os.makedirs(_output_folder)

    def _video_to_frames(self, video_path: str) -> None:
        vidcap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_jpg_folder = os.path.join(self._jpg_folder, video_name)
        video_tensor_path = os.path.join(self._tensor_folder, f"{video_name}.pt")
        self._create_folder(video_jpg_folder)

        if not vidcap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        if not os.path.exists(video_jpg_folder):
            os.makedirs(video_jpg_folder)

        frame_count = 0
        frames_list = []

        while True:
            success, image = vidcap.read()
            if not success:
                break

            # NOTE: if it is desired that the length of each file name is the same, then the following line should be modified
            frame_filename = os.path.join(
                video_jpg_folder, f"frame{frame_count:01d}.jpg"
            )
            cv2.imwrite(frame_filename, image)

            frames_tensor = transforms.ToTensor()(image)
            frames_list.append(frames_tensor)

            frame_count += 1

        vidcap.release()
        frames_tensor = torch.stack(frames_list)
        torch.save(frames_tensor, video_tensor_path)
        print(f"Done! Extracted {frame_count} frames to {video_jpg_folder}")


def main():
    video_path = r"data\raw"
    output_folder = r"data\processed"

    vtf = VideoToFrames(video_path, output_folder)
    vtf.process_videos_in_directory()


if __name__ == "__main__":
    main()
