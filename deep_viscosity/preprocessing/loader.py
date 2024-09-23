import os
from typing import Tuple
from _typeshed import StrPath

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from dataset import DeepViscosityDataset
from utils.transforms import transform


def create_dataloaders(
    batch_size: int,
    processed_data: StrPath,
    validation_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the training, testing and validation data loaders for training
    regression models.

    Args:
        batch_size (int): batch size used for training the model
        processed_data (StrPath): path to processed folder
        validation_size (float, optional): validation size. Defaults to 0.2.
        test_size (float, optional): test size. Defaults to 0.2.
    """
    viscosity_groups = {}

    # filename structure: [viscosity value]_[video number]
    for filename in os.listdir(processed_data):
        viscosity = float(filename.split("_")[0])
        if viscosity not in viscosity_groups:
            viscosity_groups[viscosity] = []
        # viscosity groups structure: {viscosity: [video1, video2, ..., videoN]}
        viscosity_groups[viscosity].append(filename)

    unique_viscosities = np.array(list(viscosity_groups.keys()))
    X_temp, X_test = train_test_split(unique_viscosities, test_size=test_size)
    X_train, X_val = train_test_split(X_temp, test_size=validation_size)

    train_folders = get_viscosity_folders(X_train, viscosity_groups)
    val_folders = get_viscosity_folders(X_val, viscosity_groups)
    test_folders = get_viscosity_folders(X_test, viscosity_groups)

    transform_function = transform()

    train_set = DeepViscosityDataset(
        processed_data, train_folders, transform=transform_function
    )
    val_set = DeepViscosityDataset(
        processed_data, val_folders, transform=transform_function
    )
    test_set = DeepViscosityDataset(
        processed_data, test_folders, transform=transform_function
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


def get_viscosity_folders(
    viscosities: list[str], viscosity_groups: dict[float : list[str]]
) -> list[str]:
    """Gets the all folders for a given list of viscosities.

    Args:
        unique_viscosities (list[str]): List of viscosities to get.
        viscosity_groups (dict[float : list[str]]): Dictionary containing the viscosity groups.

    Returns:
        list[str]: List of files for the given viscosities.
    """
    files = []
    for viscosity in viscosities:
        files.extend(viscosity_groups[viscosity])
    return files
