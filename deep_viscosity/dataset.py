import os
import re
import torch
import PIL
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

import sklearn.model_selection as ms

import utils.transforms as T


class Dataset:
    def __init__(
        self,
        processed_data_path: str,
        batch_size: int,
        test_size: float = 1 / 3,
        validation_size: float = 1 / 3,
        transform: transforms.Compose = T.transform(),
    ):
        """Create a dataset object that can be used to create dataloaders for training, validation, and testing.

        Args:
            data_path (str): Path to the directory containing the frames.
            batch_size (int): Batch size for the dataloaders.
            test_size (float, optional): Fraction of total data used for test. Defaults to 1/3.
            validation_size (float, optional): Fraction of total data used for validation. Defaults to 1/2.
        """
        self._processed_data_path = processed_data_path
        self._batch_size = batch_size
        self._test_size = test_size
        self._validation_size = (1 / (1 - test_size)) * validation_size
        self._transform = transform
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

        for folder in tqdm(os.listdir(self._processed_data_path)):
            viscosity = re.search(r"(.*)_\d+", folder).group(1)
            labels.append(float(viscosity))
            x_out = []
            for frame in os.listdir(os.path.join(self._processed_data_path, folder)):
                # filenames have the format float_testnumber.pt
                # we want to extract the float part
                image = PIL.Image.open(
                    os.path.join(self._processed_data_path, folder, frame)
                ).convert("L")
                transformed_image = self._transform(image)
                x_out.append(transformed_image.squeeze_(0))
            tensors.append(torch.stack(x_out))

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


def main():
    tensor__processed_data_path = os.path.join("data", "processed", "tensor")

    dataset = Dataset(tensor__processed_data_path, 10)
    train_loader, val_loader, test_loader = dataset.create_dataloaders()

    # Iterate through DataLoader
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print("Features:", features.shape)
        print("Labels:", labels)
    return


if __name__ == "__main__":
    main()
