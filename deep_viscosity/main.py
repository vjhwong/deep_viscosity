import os
from preprocessing.loader import create_dataloaders
from modelling.model import CNN3DVisco
from modelling.train import train
from modelling.test import test
import torch


torch.manual_seed(0)

def main() -> None:
    torch.cuda.empty_cache()
    processed_data_path = os.path.join("data", "processed")
    train_loader, test_loader, valid_loader = create_dataloaders(
        batch_size=16,
        processed_data_path=processed_data_path,
        validation_size=0.15,
        test_size=0.15,
    )

    model = CNN3DVisco(55, 210, 220)

    train(model, train_loader, valid_loader, 0.001, 20)

    # model.load_state_dict(state_dict=torch.load("trained_model.pth", weights_only=True))
    # model.eval()

    # test(model, test_loader)
    return


if __name__ == "__main__":
    main()
