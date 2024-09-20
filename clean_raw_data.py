import os


def get_video_index(video_name: str):
    return int(video_name.split("_")[1].split(".")[0])


def get_new_video_name(video_name: str):
    video_index = get_video_index(video_name) - 1

    return f"{video_name.split('_')[0]}__{video_index}.avi"


def main():
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


if __name__ == "__main__":
    main()
