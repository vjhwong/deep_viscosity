import os
import re
from typing import Tuple

import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from dataset import DeepViscosityDataset
from utils.transforms import transform


def create_reg_datasets(
    batch_size: int,
    processed_data_path: str,
    validation_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the training, testing and validation data loaders for training
    regression models. Note that the each training video is stored inside a
    folder with the folder named with its corresponding viscosity value.

    Args:
        batch_size (int): batch size used for training the model
        processed_data_path (str): path to processed folder
        validation_size (float, optional): validation size. Defaults to 0.2.
        test_size (float, optional): test size. Defaults to 0.2.
    """
    x_list = []
    y_list = []

    for filename in os.listdir(processed_data_path):
        viscosity = re.search(r"^[\d\.]+(?=_)", filename)
        if viscosity:
            viscosity = viscosity.group()
            x_list.append(viscosity)
            y_list.append(float(viscosity))

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(
        x_list, y_list, test_size=test_size, random_state=0
    )
    transform_function = transform()
    train_set = DeepViscosityDataset(
        processed_data_path, train_list, train_label, transform=transform_function
    )
    test_set = DeepViscosityDataset(
        processed_data_path, test_list, test_label, transform=transform_function
    )

    # split into training and validation batches
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    return train_loader, test_loader, valid_loader
