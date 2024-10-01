from remove_first_video import remove_first_video
from rename_data_files import rename_videos
from mask_frames import mask_videos
from frame_selection import frame_selection


def main():
    #data paths
    raw_data_path = "data/raw"
    raw_data_modified_path = "data/raw_modified"
    masked_data_path = "data/masked"
    processed_data_path = "data/processed"

    #remove first video
    remove_first_video(raw_data_path, raw_data_modified_path)
    
    #rename data files
    rename_videos(raw_data_modified_path)
    
    #mask frames
    mask_path = "data/masks.npy"
    mask_videos(raw_data_modified_path, mask_path)
    
    #frame selection
    frame_range = (45, 99)
    frame_selection(processed_data_path, masked_data_path, frame_range)

if __name__ == "__main__":
    main()