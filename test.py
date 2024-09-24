import os
from deep_viscosity.modelling import train
from deep_viscosity.preprocessing import loader
from deep_viscosity.modelling.model import CNN3DVisco
import torch

torch.manual_seed(42)

processed_data_path = os.path.join("data", "processed")
train_loader, test_loader, valid_loader = loader.create_dataloaders(
    batch_size=1, processed_data_path=processed_data_path
)

model = CNN3DVisco(55, 210, 220)

input = train.train(model, train_loader, valid_loader, 0.01, 1)
print(input)
