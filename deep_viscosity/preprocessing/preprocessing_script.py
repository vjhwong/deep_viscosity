
import os
from clean_raw_data import get_new_video_name, get_video_index
from rename_data_files import find_interpolated_viscosities, rename_videos
from mask_frames import mask_frames, mask_videos, get_video_name
from frame_selection import frame_selection


def main():
    #clean_raw_data
    video_folder_path = os.path.join("data", "raw")

    for video_name in sorted(os.listdir(video_folder_path), reverse=False):
        video_path = os.path.join(video_folder_path, video_name)
        video_index = get_video_index(video_name)

        if video_index == 1:
            os.remove(video_path)
        else:
            new_video_name = get_new_video_name(video_name)
            new_video_path = os.path.join(video_folder_path, new_video_name)
            os.rename(video_path, new_video_path)
    
    #rename data files
    percentages, viscosities = find_interpolated_viscosities()
    rename_videos(percentages, viscosities)
    
    #mask frames
    mask_path = os.path.join("data", "masks.npy")

    mask_videos(video_folder_path, mask_path)
    
    output_path = "data/processed"
    input_path = "data/masked"
    frame_range = (45, 99)
    
    #frame selection
    frame_selection(output_path, input_path, frame_range)
    


if __name__ == "__main__":
    main()