import sys
import os
import argparse

from deep_viscosity.loader import create_dataloaders
from deep_viscosity.modelling.model import CNN3DVisco
from deep_viscosity.modelling.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 3D CNN model for viscosity prediction."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the validation split",
    )

    args = parser.parse_args()

    # Load dataset
    train_loader, val_loader = create_dataloaders(
        args.data_path, args.batch_size, args.test_size, args.val_size
    )

    # Initialize model
    model = CNN3DVisco()

    # Train model
    train(model, train_loader, val_loader, args.num_epochs, args.learning_rate)


if __name__ == "__main__":
    main()
