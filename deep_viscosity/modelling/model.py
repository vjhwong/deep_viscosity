import torch
import torch.nn as nn
import torch.nn.functional as func
import modelling.utils.functions as f


class DeepViscosityModel(nn.Module):  # här nere får vi ändra sen
    def __init__(
        self,
        t_dim: int,
        img_x: int,
        img_y: int,
        # dropout: float = 0.1,
        fc_hidden1: int = 256,
        fc_hidden2: int = 100,
    ) -> None:
        """Create a 3D CNN model for predicting viscosity.

        Args:
            t_dim (int): Frame dimension.
            img_x (int): Resolution in x.
            img_y (int): Resolution in y.
            dropout (float): Dropout rate. Defaults to 0.
            fc_hidden1 (int, optional): Number of nodes in first fully-connected layer. Defaults to 256.
            fc_hidden2 (int, optional): Number of nodes in first fully-connected layer. Defaults to 256.
        """
        super().__init__()
        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        # self.dropout = dropout
        self.ch1, self.ch2 = 70, 80
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        # compute conv1 & conv2 output shape
        self.conv1_outshape = f.conv3d_output_size(
            (self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1
        )
        self.conv2_outshape = f.conv3d_output_size(
            self.conv1_outshape, self.pd2, self.k2, self.s2
        )
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.ch1,
            kernel_size=self.k1,
            stride=self.s1,
            padding=self.pd1,
        )
        # self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(
            in_channels=self.ch1,
            out_channels=self.ch2,
            kernel_size=self.k2,
            stride=self.s2,
            padding=self.pd2,
        )
        # self.bn2 = nn.BatchNorm3d(self.ch2)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        # self.drop = nn.Dropout3d(self.dropout)
        self.pool = nn.MaxPool3d(2)
        # fully connected hidden layer
        self.fc1 = nn.Linear(
            self.ch2
            * self.conv2_outshape[0]
            * self.conv2_outshape[1]
            * self.conv2_outshape[2],
            self.fc_hidden1,
        )
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # fully connected layer, output = multi-classes
        self.fc3 = nn.Linear(self.fc_hidden2, 1)
        nn.init.constant_(self.fc3.bias, 470)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_3d (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Conv 1
        x_out = self.conv1(x_3d)
        # x_out = self.bn1(x_out)
        x_out = self.leakyrelu(x_out)
        # x_out = self.drop(x_out)
        # Conv 2
        x_out = self.conv2(x_out)
        # x_out = self.bn2(x_out)
        x_out = self.leakyrelu(x_out)
        # x_out = self.drop(x_out)
        # flatten the conv2 to feed to fc layers
        x_out = x_out.view(x_out.size(0), -1)

        # FC 1 and 2
        x_out = func.leaky_relu(self.fc1(x_out))
        x_out = func.leaky_relu(self.fc2(x_out))

        # removes neurons randomly while training
        # x_out = func.dropout(x_out, p=self.dropout, training=self.training)

        x_out = self.fc3(x_out)
        return x_out
