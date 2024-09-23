import os
from typing import Tuple
from _typeshed import StrPath

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from preprocessing.dataset import DeepViscosityDataset
from preprocessing.utils.transforms import transform
from preprocessing.utils.enums import ViscosityClass


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
    viscosity_df = pd.DataFrame(columns=["folder_name", "classification"])

    # folder_name structure: [viscosity value]_[video number]
    for folder_name in os.listdir(processed_data):
        viscosity = float(folder_name.split("_")[0])
        viscosity_df = viscosity_df.append(
            {
                "folder_name": folder_name,
                "classification": get_classification(viscosity),
            },
            ignore_index=True,
        )

    split_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )
    for train_val_index, test_index in split_test.split(
        viscosity_df, viscosity_df["classification"]
    ):
        X_train_val = viscosity_df.loc[train_val_index]
        X_test = viscosity_df.loc[test_index]

    split_val = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=42
    )
    for train_index, val_index in split_val.split(
        X_train_val, X_train_val["classification"]
    ):
        X_train = X_train_val.loc[train_index]
        X_val = X_train_val.loc[val_index]

    train_folders = X_train["folder_name"].tolist()
    val_folders = X_val["folder_name"].tolist()
    test_folders = X_test["folder_name"].tolist()

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
