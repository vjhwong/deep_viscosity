import argparse
import random
import numpy as np
import torch

from modelling.model import DeepViscosityModel
from preprocessing.loader import create_dataloaders
from modelling.train import train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(0)

    parser = argparse.ArgumentParser(
        description="Train a 3D CNN model for viscosity prediction."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )

    parser.add_argument(
        "x_dim", type=int, required=True,help="Resolution in x dimension of the input data"
    )

    parser.add_argument(
        "y_dim", type=int, required=True, help="Resolution in y dimension of the input data"
    )

    parser.add_argument(
        "t_dim", type=int, required=True, help="Resolution in t dimension of the input data"
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
        default=0.15,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Proportion of the dataset to include in the validation split",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for loading the data",
    )

    args = parser.parse_args()

    # Load dataset
    train_loader, val_loader = create_dataloaders(
        args.data_path, args.batch_size, args.test_size, args.val_size, args.num_workers
    )

    # Initialize model
    model = DeepViscosityModel(args.t_dim, args.x_dim, args.y_dim)

    # Train model
    train(model, train_loader, val_loader, args.num_epochs, args.learning_rate)


if __name__ == "__main__":
    main()
