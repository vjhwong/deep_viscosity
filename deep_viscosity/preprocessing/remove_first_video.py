import os
import shutil


def get_video_index(video_name: str):
    return int(video_name.split("_")[1].split(".")[0])


def get_new_video_name(remove_first_vid, video_name: str):
    if remove_first_vid:
        video_index = get_video_index(video_name) - 1
    else:
        video_index = get_video_index(video_name)

    return f"{video_name.split('_')[0]}__{video_index}.avi"

def remove_first_video(remove_first_vid, input_path, output_path):
    if not os.path.exists(output_path):
        shutil.copytree(input_path, output_path)
    else:
        print(f"The {output_path} already exists!")
        return

    for video_name in sorted(os.listdir(output_path), reverse=False):
        video_path = os.path.join(output_path, video_name)
        video_index = get_video_index(video_name)

        if video_index == 1 and remove_first_vid:
            os.remove(video_path)
        else:
            new_video_name = get_new_video_name(remove_first_vid, video_name)
            new_video_path = os.path.join(output_path, new_video_name)
            os.rename(video_path, new_video_path)

def main():
    input_path = "data/raw"
    output_path = "data/raw_modified"
    remove_first_video(input_path, output_path)

if __name__ == "__main__":
    main()
