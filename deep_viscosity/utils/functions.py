<<<<<<< HEAD
import numpy as np

def conv3d_output_size(img_size, padding, kernel_size, stride):
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
=======
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch

def conv3d_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] -
                          (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] -
                          (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] -
                          (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape
>>>>>>> a689167 (feat: Added a script that creates the model, a scirpt that test and a script that trains.)
