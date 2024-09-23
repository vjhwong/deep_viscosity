import os
from preprocessing.loader import create_dataloaders
from modelling.model import CNN3DVisco
from modelling.train import train
from modelling.test import test

def main():
    processed_data_path = os.path.join("data", "processed")
    train_loader, test_loader, valid_loader = create_dataloaders(
        batch_size=8, processed_data_path=processed_data_path
    )

    model = CNN3DVisco(55, 210, 220)

    train(model, train_loader, valid_loader, 0.05, 2)
    
    test("path_to_model",test_loader)
    
    return


if __name__ == "__main__":
    main()
