import os
import shutil

output_path = "data/processed"
input_path = "data/masked"

def frame_selection(output_path: str, input_path: str):
  if not os.path.exists(output_path):
  os.makedirs(output_path)

  for file in os.listdir(input_path):
  print(file)
    # shutil.copy(source, destination)