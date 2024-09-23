import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DeepViscosityDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        processed_data_path: str,
        folder_list: list[str],
        transform: transforms.Compose = None,
    ):
        """Initializes the DeepViscosityDataset class.

        Args:
            processed_data_path (str): Path to the directory containing the processed data.
            folder_list (list[str]): List of files to use for the dataset.
            transform (transforms.Compose, optional): Transforms used on images. Defaults to None.
        """
        self.processed_data_path = processed_data_path
        self.folder_list = folder_list
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folder_list)

    def read_images(self, folder: str, transform: transforms.Compose) -> torch.Tensor:
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
            image_path = os.path.join(self.processed_data_path, folder, image)
            image = Image.open(image_path).convert("L")

            if transform is not None:
                image = transform(image)

            video_tensor.append(image.squeeze_(0))

        return torch.stack(video_tensor, dim=0)

    def get_viscosity(self, folder: str) -> float:
        """Gets the viscosity value from a folder name.

        Args:
            folder (str): Folder name containing the viscosity value.

        Returns:
            float: Viscosity value.
        """
        return float(folder.split("_")[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates one sample of data.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the data tensor and label tensor.
        """
        folder = self.folder_list[index]
        data_tensor = self.read_images(folder, self.transform).unsqueeze_(0)
        label_tensor = torch.FloatTensor(self.get_viscosity(folder))
        return data_tensor, label_tensor
