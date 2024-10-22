import argparse
import random
import numpy as np
import torch
import glob
import shutil

from modelling.model import DeepViscosityModel
from preprocessing.loader import create_dataloaders
from modelling.test import test


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Test a 3D CNN model for viscosity prediction."
    )

    parser.add_argument(
        "--random_seed", type=int, help="Random seed"
    )

    parser.add_argument(
        "--model_path", type=str, help="Path to the trained model"
    )

    parser.add_argument(
        "--data_path", type=str, help="Path to the dataset"
    )

    parser.add_argument(
        "--x_dim", type=int,help="Resolution in x dimension of the input data"
    )

    parser.add_argument(
        "--y_dim", type=int, help="Resolution in y dimension of the input data"
    )

    parser.add_argument(
        "--t_dim", type=int, help="Resolution in t dimension of the input data"
    )

    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training"
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
    set_seed(args.random_seed)

    # Save slurm output
    slurm_files = glob.glob("slurm*")
    for idx, slurm_file in enumerate(slurm_files):
        destination_path = f"{args.model_path.split('.')[0]}_{idx}_test.out" if len(slurm_files) > 1 else f"{args.model_path.split('.')[0]}_test.out"
        shutil.move(slurm_file, destination_path)
        print(f"Moved and renamed {slurm_file} to {destination_path}")

    # Load dataset
    train_loader, test_loader, val_loader = create_dataloaders(
        args.batch_size, args.data_path, args.random_seed, args.test_size, args.val_size, args.num_workers
    )

    # Initialize model
    model = DeepViscosityModel(args.t_dim, args.x_dim, args.y_dim)

    # Load model
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()    

    # Train model
    (figure, test_targets, test_outputs) = test(model, test_loader, train_loader, val_loader, plot_all=True)
    figure.savefig(args.model_path.split(".")[0] + "_testplot.png")
    np.save(args.model_path.split(".")[0] + "_testtargets", np.array(test_targets))
    np.save(args.model_path.split(".")[0] + "_testpredictions", np.array(test_outputs))


if __name__ == "__main__":
    main() 
