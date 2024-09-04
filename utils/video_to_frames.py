import cv2
import os


class VideoToFrames:
    def __init__(self, data_path: str, output_folder: str) -> None:
        self._data_path = data_path
        self._output_folder = output_folder
        self._create_folder(self._output_folder)

    ### Public methods ###

    def process_videos_in_directory(self) -> None:
        video_files = [
            file for file in os.listdir(self._data_path) if file.endswith(".mp4")
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
        frames_folder = os.path.join(self._output_folder, os.path.basename(video_path))

        if not vidcap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        frame_count = 0

        while True:
            success, image = vidcap.read()
            if not success:
                break

            # NOTE: if it is desired that the length of each file name is the same, then the following line should be modified
            frame_filename = os.path.join(frames_folder, f"frame{frame_count:01d}.jpg")

            cv2.imwrite(frame_filename, image)

            frame_count += 1

        vidcap.release()
        print(f"Done! Extracted {frame_count} frames to {frames_folder}")


def main():
    video_path = r"data\raw"
    output_folder = r"data\processed"
    vtf = VideoToFrames(video_path, output_folder)
    vtf.process_videos_in_directory()


if __name__ == "__main__":
    main()
