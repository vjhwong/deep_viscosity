from PIL import Image
import os


def get_zeroed_data(input_folder: str, output_folder: str):
    """Creates black videos with the same dimensions as the original data.

    Args:
        input_folder (str): Path to the directory containing the original video files.
        output_folder (str): Path to the directory where the black video data will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_folder in os.listdir(input_folder):
        os.makedirs(os.path.join(output_folder, video_folder))
        for image in os.listdir(os.path.join(input_folder, video_folder)):
            input_directory = os.path.join(input_folder, video_folder, image)
            output_directory = os.path.join(output_folder, video_folder, image)
            convert_to_black(input_directory, output_directory)


def convert_to_black(image_path: str, output_path: str):
    """Creates a black with the same dimensions as the input image.

    Args:
        image_path (str): Path to the original_image.
        output_path (str): Path to the directory where the black image will be saved.
    """
    img = Image.open(image_path)
    black_image = Image.new("RGB", img.size, (0, 0, 0))
    black_image.save(output_path)


def main():
    input_path = "data/processed"
    output_path = "data/black"

    get_zeroed_data(input_path, output_path)


if __name__ == "__main__":
    main()
