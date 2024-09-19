import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DeepViscosityDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        processed_data_path: str,
        train_folders: list[str],
        train_labels: str,
        transform: transforms.Compose = None,
    ):
        """Initializes the DeepViscosityDataset class.

        Args:
            processed_data_path (str): Path to the directory containing the processed data.
            train_folders (list[str]): List of folders containing the training data.
            train_labels (str): List of labels for the training data.
            transform (transforms.Compose, optional): Transforms used on images. Defaults to None.
        """
        self.processed_data_path = processed_data_path
        self.train_folders = train_folders
        self.train_labels = train_labels
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.train_folders)

    def read_images(
        self, path: str, folder: str, transform: transforms.Compose
    ) -> torch.Tensor:
        """Reads images from a folder and applies the specified transforms.

        Args:
            path (str): Path to the directory containing the images.
            folder (str): Folder containing the images.
            transform (transforms.Compose): Transforms to apply to the images.

        Returns:
            torch.Tensor: Tensor containing the images.
        """
        video_tensor = []
        for image in os.listdir(os.path.join(self.processed_data_path, folder)):
            image = Image.open(
                os.path.join(self.processed_data_path, folder, image)
            ).convert("L")

            if transform is not None:
                image = transform(image)

            video_tensor.append(image.squeeze_(0))

        video_tensor = torch.stack(video_tensor, dim=0)

        return video_tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates one sample of data.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the video tensor and label tensor.
        """
        folder = self.train_folders[index]
        video_tensor = self.read_images(
            self.processed_data_path, folder, self.transform
        ).unsqueeze_(0)
        label_tensor = torch.FloatTensor([self.train_labels[index]])
        return video_tensor, label_tensor
