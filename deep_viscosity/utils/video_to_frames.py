import cv2
import os
import torch
import numpy as np
from torchvision import transforms


class VideoCaptureError(Exception):
    pass


class VideoToFrames:
    def __init__(self, raw_data_path: str, output_folder: str, n_frames: int) -> None:
        """Initializes the VideoToFrames class.

        Args:
            raw_data_path (str): Path to the directory containing the video files.
            output_folder (str): Path to the directory where the frames will be saved.
            n_frames (int): Number of frames to extract from each video.
        """
        self._raw_data_path = raw_data_path
        self._output_folder = output_folder
        self._n_frames = n_frames

        self._create_folder(self._output_folder)

    ### Public methods ###

    def process_videos_in_directory(self) -> None:
        """
        Processes all the video files in the directory.
        """
        video_files = [
            file for file in os.listdir(self._raw_data_path) if file.endswith(".avi")
        ]
        for video_file in video_files:
            video_path = os.path.join(self._raw_data_path, video_file)
            self._video_to_frames(video_path)

    ### Private methods ###

    def _video_to_frames(self, video_path: str) -> None:
        """Extracts frames from a video file and saves them as jpg files.

        Args:
            video_path (str): Path to the video file.
        """
        vidcap = cv2.VideoCapture(video_path)
        if not self._check_vidcap(vidcap):
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_jpg_folder = os.path.join(self._output_folder, video_name)
        self._create_folder(video_jpg_folder)

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frames = np.linspace(40, total_frames - 20, self._n_frames, dtype=int)

        for frame_index in range(total_frames):
            success, frame = vidcap.read()

            if not success:
                break
            if frame_index not in selected_frames:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # NOTE: if it is desired that the length of each file name is the same, then the following line should be modified
            frame_filename = os.path.join(
                video_jpg_folder, f"frame{frame_index:01d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)

        vidcap.release()
        print(f"Done! Extracted {self._n_frames} frames to {video_jpg_folder}")

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
    raw_data_path = os.path.join("data", "raw")
    processed_data_path = os.path.join("data", "processed")
    video_to_frames = VideoToFrames(raw_data_path, processed_data_path, n_frames=20)
    video_to_frames.process_videos_in_directory()


if __name__ == "__main__":
    main()
