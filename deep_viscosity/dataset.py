import os
import re
import torch

import sklearn.model_selection as ms

from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, data_path: str, batch_size: int):
        self._data_path = data_path
        self._batch_size = batch_size
        (
            self._X_train,
            self._X_val,
            self._X_test,
            self._y_train,
            self._y_val,
            self._y_test,
        ) = self._split_data()

    def _split_data(self) -> tuple[list]:
        tensors = []
        labels = []

        for tensor in os.listdir(self._data_path):
            viscosity = re.search(r"\d+", tensor).group(0)
            tensors.append(torch.load(os.path.join(self._data_path, tensor)))
            labels.append(viscosity)

        X_train, X_test, y_train, y_test = ms.train_test_split(
            tensors, labels, test_size=1 / 3, random_state=42
        )
        X_train, X_val, y_train, y_val = ms.train_test_split(
            X_train, y_train, test_size=1 / 2, random_state=42
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(self) -> tuple[DataLoader]:
        print(self._X_train)
        X_train_tensor = torch.tensor(self._X_train)
        y_train_tensor = torch.tensor(self._y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        X_val_tensor = torch.tensor(self._X_val)
        y_val_tensor = torch.tensor(self._y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=True)

        X_test_tensor = torch.tensor(self._X_test)
        y_test_tensor = torch.tensor(self._y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=True
        )
        return train_loader, val_loader, test_loader
