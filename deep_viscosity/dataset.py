import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DeepViscosityDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, processed_data_path: str, train_folders: list[str], train_labels: str, transform: transforms.Compose = None):
        "Initialization"
        self.processed_data_path = processed_data_path
        self.train_folders = train_folders
        self.train_labels = train_labels
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.train_folders)

    def read_images(self, path: str, folder: str, transform: transforms.Compose):
        video_tensor = []
        for image in os.listdir(path.join(self.processed_data_path, folder)):
            image = Image.open(path.join(self.processed_data_path, folder, image)).convert('L')

            if transform is not None:
                image = transform(image)

            video_tensor.append(image.squeeze_(0))

        video_tensor = torch.stack(video_tensor, dim=0)

        return video_tensor

    def __getitem__(self, index):
        "Generates one sample of data"

        folder = self.train_folders[index]
        video_tensor = self.read_images(self.processed_data_path, folder, self.transform).unsqueeze_(0)
        label_tensor = torch.LongTensor([self.train_labels[index]])

        return video_tensor, label_tensor
