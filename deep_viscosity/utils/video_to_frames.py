import cv2
import os
import torch
from torchvision import transforms


class VideoCaptureError(Exception):
    pass


class VideoToFrames:
    def __init__(self, data_path: str, output_folder: str) -> None:
        """Initializes the VideoToFrames class.

        Args:
            data_path (str): Path to the directory containing the video files.
            output_folder (str): Path to the directory where the frames will be saved.
        """
        self._data_path = data_path
        self._output_folder = output_folder
        self._jpg_folder = os.path.join(self._output_folder, "jpg")
        self._tensor_folder = os.path.join(self._output_folder, "tensor")
        self._create_folder(self._jpg_folder)
        self._create_folder(self._tensor_folder)

    ### Public methods ###

    def process_videos_in_directory(self) -> None:
        """
        Processes all the video files in the directory.
        Saves the frames as jpg files and the frames as tensors.
        """
        video_files = [
            file for file in os.listdir(self._data_path) if file.endswith(".avi")
        ]
        for video_file in video_files:
            video_path = os.path.join(self._data_path, video_file)
            self._video_to_frames(video_path)

    ### Private methods ###

    def _video_to_frames(self, video_path: str) -> None:
        """Extracts frames from a video file and saves them as jpg files and as tensors.

        Args:
            video_path (str): Path to the video file.
        """
        vidcap = cv2.VideoCapture(video_path)
        if not self._check_vidcap(vidcap):
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_jpg_folder = os.path.join(self._jpg_folder, video_name)
        video_tensor_path = os.path.join(self._tensor_folder, f"{video_name}.pt")
        self._create_folder(video_jpg_folder)

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

    def _create_folder(self, _output_folder: str) -> None:
        """Creates a folder if it does not aldready exist.

        Args:
            _output_folder (str): Path to the folder that will be created.
        """

        if not os.path.exists(_output_folder):
            os.makedirs(_output_folder)

    def _check_vidcap(self, vidcap: cv2.VideoCapture) -> bool:
        """Checks if a video capture object is opened.

        Args:
            vidcap (cv2.VideoCapture): A video capture object.

        Returns:
            bool: True if the video capture object is opened.

        Raises:
            VideoCaptureError: If the video capture object is not opened.
        """
        if not vidcap.isOpened():
            raise VideoCaptureError("Error opening video file.")
        return True


def main():
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    video_to_frames = VideoToFrames(raw_data_path, processed_data_path)
    video_to_frames.process_videos_in_directory()


if __name__ == "__main__":
    main()
