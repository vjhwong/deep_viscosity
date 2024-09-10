import os
import re
import torch

import sklearn.model_selection as ms

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Dataset:
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        test_size: float = 1 / 3,
        validation_size: float = 1 / 3,
    ):
        """Create a dataset object that can be used to create dataloaders for training, validation, and testing.

        Args:
            data_path (str): Path to the directory containing the tensor data.
            batch_size (int): Batch size for the dataloaders.
            test_size (float, optional): Fraction of total data used for test. Defaults to 1/3.
            validation_size (float, optional): Fraction of total data used for validation. Defaults to 1/2.
        """
        self._data_path = data_path
        self._batch_size = batch_size
        self._test_size = test_size
        self._validation_size = (1 / (1 - test_size)) * validation_size

        (
            self._X_train,
            self._X_val,
            self._X_test,
            self._y_train,
            self._y_val,
            self._y_test,
        ) = self._split_data()

    def _split_data(self) -> tuple[list]:
        """Split the data into training, validation, and testing sets.

        Returns:
            tuple[list]: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        tensors = []
        labels = []

        for tensor in tqdm(os.listdir(self._data_path)):
            # filenames have the format float_testnumber.pt
            # we want to extract the float part
            viscosity = re.search(r"(.*)_\d+", tensor).group(1)
            tensors.append(torch.load(os.path.join(self._data_path, tensor)))
            labels.append(float(viscosity))

        X_train, X_test, y_train, y_test = ms.train_test_split(
            tensors, labels, test_size=self._test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = ms.train_test_split(
            X_train, y_train, test_size=self._validation_size, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self) -> tuple[DataLoader]:
        """Create dataloaders for training, validation, and testing.

        Returns:
            tuple[DataLoader]: Train, validation, and test dataloaders.
        """
        X_train_tensor = torch.stack(self._X_train)
        y_train_tensor = torch.tensor(self._y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        X_val_tensor = torch.stack(self._X_val)
        y_val_tensor = torch.tensor(self._y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True)

        X_test_tensor = torch.stack(self._X_test)
        y_test_tensor = torch.tensor(self._y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=True
        )
        return train_loader, val_loader, test_loader
