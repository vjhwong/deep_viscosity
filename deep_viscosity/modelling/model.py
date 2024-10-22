import torch
import torch.nn as nn
import torch.nn.functional as func

import modelling.utils.functions as f


class DeepViscosityModel(nn.Module):
    def __init__(
        self,
        t_dim: int,
        img_x: int,
        img_y: int,
    ) -> None:
        """Create a 3D CNN model for predicting viscosity.

        Args:
            t_dim (int): Frame dimension.
            img_x (int): Resolution in x.
            img_y (int): Resolution in y.
        """
        super().__init__()

        # set dimensions
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y

        # convolutional layers, kernel size, stride & padding
        self.ch1, self.ch2, self.ch3 = 20, 40, 60
        self.k1, self.k2, self.k3 = (
            3, 3, 3), (3, 3, 3), (3, 3, 3)
        self.s1, self.s2, self.s3 = (
            2, 2, 2), (2, 2, 2), (2, 2, 2)
        self.pd1, self.pd2, self.pd3 = (
            0, 0, 0), (0, 0, 0), (0, 0, 0)
        
        # output shape of convolutional layers
        self.conv1_outshape = f.conv3d_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1
        )
        self.conv2_outshape = f.conv3d_output_size(
            self.conv1_outshape, self.pd2, self.k2, self.s2
        )
        self.conv3_outshape = f.conv3d_output_size(
            self.conv2_outshape, self.pd3, self.k3, self.s3
        )

        # convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.ch1,
            kernel_size=self.k1,
            stride=self.s1,
            padding=self.pd1,
        )
        self.conv2 = nn.Conv3d(
            in_channels=self.ch1,
            out_channels=self.ch2,
            kernel_size=self.k2,
            stride=self.s2,
            padding=self.pd2,
        )
        self.conv3 = nn.Conv3d(
            in_channels=self.ch2,
            out_channels=self.ch3,
            kernel_size=self.k3,
            stride=self.s3,
            padding=self.pd3,
        )

        # fully connected layers
        fc_hidden1, fc_hidden2 = 256, 100
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2

        self.fc1 = nn.Linear(
            self.ch3
            * self.conv3_outshape[0]
            * self.conv3_outshape[1]
            * self.conv3_outshape[2],
            self.fc_hidden1,
        )
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, 1)
        nn.init.constant_(self.fc3.bias, 470)

        # leaky ReLu activation function
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_3d (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # convolutional layer 1
        x_out = self.conv1(x_3d)
        x_out = self.leakyrelu(x_out)

        # convolutional layer 2
        x_out = self.conv2(x_out)
        x_out = self.leakyrelu(x_out)

        # convolutional layer 3
        x_out = self.conv3(x_out)
        x_out = self.leakyrelu(x_out)

        # flatten
        x_out = x_out.view(x_out.size(0), -1)

        # fully connected layers
        x_out = func.leaky_relu(self.fc1(x_out))
        x_out = func.leaky_relu(self.fc2(x_out))

        x_out = self.fc3(x_out)
        return x_out
