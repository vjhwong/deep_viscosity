from remove_first_video import remove_first_video
from rename_data_files import rename_videos
from mask_frames import mask_videos
from frame_selection import frame_selection
from utils.preprocess_args import get_args


def main():
    "The entire data preprocessing pipeline."

    # data paths
    raw_data_path = "data/raw"
    raw_data_modified_path = "data/raw_modified"
    masked_data_path = "data/masked"
    processed_data_path = "data/processed"

    # potentially remove first video
    remove_first_video(args.remove_first_vid, raw_data_path, raw_data_modified_path)

    # rename data files
    rename_videos(raw_data_modified_path)

    # mask frames
    mask_path = args.mask_path
    mask_videos(raw_data_modified_path, masked_data_path, mask_path)

    # frame selection
    frame_range = (args.first_frame, args.last_frame)
    frame_selection(processed_data_path, masked_data_path, frame_range)


if __name__ == "__main__":
    args = get_args()
    main()
