import utils.functions as f
from utils.preprocess_args import get_args

def main() -> None:
    """The entire data preprocessing pipeline"""
    args = get_args()
    # data paths
    raw_data_path = args.raw_data_path
    raw_data_modified_path = f.get_new_folder_path(raw_data_path, "raw_modified")
    processed_data_path = f.get_new_folder_path(raw_data_path, "processed")

    # potentially remove first video
    f.remove_first_video(args.remove_first_vid, raw_data_path, raw_data_modified_path)

    # rename data files
    f.rename_videos(raw_data_modified_path, [float(x) for x in args.percentages.split(',')])

    # mask and select frames
    mask_path = args.mask_path
    frame_range = (args.first_frame, args.last_frame)
    f.mask_videos(raw_data_modified_path, processed_data_path, mask_path, frame_range)


if __name__ == "__main__":
    main()
