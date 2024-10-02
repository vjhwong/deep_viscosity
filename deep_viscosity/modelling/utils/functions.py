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
        kernel_size (tuble[int]): The dimensions of the kernel.
        stride (tuple[int]): The stride in each dimension.

    Returns:
        tuple[int]: The dimensions of the output from the convolutional layer.
    """
    # compute output shape of conv3D
    outshape = (
        np.floor(
            (img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1
        ).astype(int),
        np.floor(
            (img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1
        ).astype(int),
        np.floor(
            (img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1
        ).astype(int),
    )
    return outshape
