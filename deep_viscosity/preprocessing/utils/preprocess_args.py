import argparse


def get_args() -> argparse.Namespace:
    """Parses command-line arguments for video preprocessing.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess a video dataset")
    parser.add_argument(
        "--raw_data_path", type=str, help="Path to raw data (video files)"
    )
    parser.add_argument(
        "--first_frame", type=int, help="Lower bound for the frame selection"
    )
    parser.add_argument(
        "--last_frame", type=int, help="Upper bound for the frame selection"
    )
    parser.add_argument("--mask_path", type=str, help="Segmentation mask, an npy-file")
    parser.add_argument(
        "--remove_first_vid",
        type=bool,
        help="Do we want the first video of each viscosity to be removed",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        help="The weight percentages of the glycerol solutions in the videos",
    )

    return parser.parse_args()
