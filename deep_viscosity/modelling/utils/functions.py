import os
import glob
import shutil

import numpy as np


def conv3d_output_size(
    img_size: tuple[int],
    padding: tuple[int],
    kernel_size: tuple[int],
    stride: tuple[int],
) -> tuple[int]:
    """Computes the output shape of a 3D convolutional layer

    Args:
        img_size (tuple[int]): The image dimensions.
        padding (tuple[int]): The desired padding in each dimension.
        kernel_size (tuple[int]): The dimensions of the kernel.
        stride (tuple[int]): The stride in each dimension.

    Returns:
        tuple[int]: The dimensions of the output from the convolutional layer.
    """
    outshape = (
        np.floor(
            (img_size[0] + 2 * padding[0] -
             (kernel_size[0] - 1) - 1) / stride[0] + 1
        ).astype(int),
        np.floor(
            (img_size[1] + 2 * padding[1] -
             (kernel_size[1] - 1) - 1) / stride[1] + 1
        ).astype(int),
        np.floor(
            (img_size[2] + 2 * padding[2] -
             (kernel_size[2] - 1) - 1) / stride[2] + 1
        ).astype(int),
    )
    return outshape


def create_run_folder(run_name: str) -> None:
    """Create a folder to store the run files.

    Args:
        run_name (str): The name of the run.
    """
    run_folder = os.path.join("models", f"{run_name}")
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    slurm_files = glob.glob("slurm*")
    for idx, slurm_file in enumerate(slurm_files, 1):
        new_file_name = f"{run_name}_{idx}.out" if len(
            slurm_files) > 1 else f"{run_name}.out"
        destination_path = os.path.join(run_folder, new_file_name)
        shutil.move(slurm_file, destination_path)
        print(f"Moved and renamed {slurm_file} to {destination_path}")
