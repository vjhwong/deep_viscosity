import os
from typing import Tuple

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from deep_viscosity.preprocessing.dataset import DeepViscosityDataset
from deep_viscosity.preprocessing.utils.transforms import transform
from deep_viscosity.preprocessing.utils.enums import ViscosityClass


def create_dataloaders(
    batch_size: int,
    processed_data_path: str,
    validation_size: float = 0.2,
    test_size: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the training, testing and validation data loaders for training
    regression models.

    Args:
        batch_size (int): batch size used for training the model
        processed_data_path (str): path to processed folder
        validation_size (float, optional): validation size. Defaults to 0.2.
        test_size (float, optional): test size. Defaults to 0.2.
    """
    # Initialize an empty list to store the data
    viscosity_data = []
    unique_viscosity_values = []

    # folder_name structure: [viscosity value]_[video number]
    for folder_name in os.listdir(processed_data_path):
        viscosity = float(folder_name.split("_")[0])
        data = {"viscosity": viscosity, "classification": get_classification(viscosity)}
        if data not in unique_viscosity_values:
            unique_viscosity_values.append(data)
        viscosity_data.append(
            {
                "folder_name": folder_name,
                "viscosity": viscosity,
            }
        )
    viscosity_df = pd.DataFrame(viscosity_data)
    unique_viscosity_df = pd.DataFrame(unique_viscosity_values)
    split_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )
    for train_val_index, test_index in split_test.split(
        unique_viscosity_df, unique_viscosity_df["classification"]
    ):
        train_val_viscosities = unique_viscosity_df.loc[train_val_index]
        test_viscosities = unique_viscosity_df.loc[test_index]

    split_val = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=42
    )
    train_val_viscosities = train_val_viscosities.reset_index(drop=True)
    for train_index, val_index in split_val.split(
        train_val_viscosities, train_val_viscosities["classification"]
    ):
        train_viscosities = train_val_viscosities.loc[train_index]
        val_viscosities = train_val_viscosities.loc[val_index]

    train_folders = viscosity_df[
        viscosity_df["viscosity"].isin(train_viscosities["viscosity"])
    ]["folder_name"].tolist()
    val_folders = viscosity_df[
        viscosity_df["viscosity"].isin(val_viscosities["viscosity"])
    ]["folder_name"].tolist()
    test_folders = viscosity_df[
        viscosity_df["viscosity"].isin(test_viscosities["viscosity"])
    ]["folder_name"].tolist()

    transform_function = transform()

    train_set = DeepViscosityDataset(
        processed_data_path, train_folders, transform=transform_function
    )
    val_set = DeepViscosityDataset(
        processed_data_path, val_folders, transform=transform_function
    )
    test_set = DeepViscosityDataset(
        processed_data_path, test_folders, transform=transform_function
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


def get_classification(viscosity: float) -> str:
    """Get the classification of the viscosity value.

    Args:
        viscosity (float): Viscosity value.

    Returns:
        str: Classification of the viscosity value.
    """
    if viscosity < 200:
        return ViscosityClass.LOW.value
    elif viscosity < 500:
        return ViscosityClass.MEDIUM.value
    else:
        return ViscosityClass.HIGH.value
