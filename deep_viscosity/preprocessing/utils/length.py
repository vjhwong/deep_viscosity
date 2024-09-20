import os

def get_max_number_of_frames(path: str):
  counts = []
  for folder in os.listdir(path):
    counter = 0
    for frame in os.listdir(os.path.join(path, folder)):
      counter +=1
    counts.append(counter)

  return max(counts)

def get_longest_video_names(max_frames, path):
  video_names = []
  for folder in os.listdir(path):
    counter = 0
    for frame in os.listdir(os.path.join(path, folder)):
      counter +=1
    if counter == max_frames:
      video_names.append(folder)
  return video_names

def main():
  path = "data/all_frames"
  max_frames = get_max_number_of_frames(path)
  longest_videos = get_longest_video_names(max_frames, path)
  print(longest_videos)
  print(max_frames)

if __name__ == "__main__":
    main()