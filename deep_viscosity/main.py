import os
from preprocessing.loader import create_reg_datasets
from modelling.model import CNN3DVisco
from modelling.train import train


def main() -> None:
    processed_data_path = os.path.join("data", "processed")
    train_loader, test_loader, valid_loader = create_reg_datasets(
        batch_size=8, processed_data_path=processed_data_path
    )

    model = CNN3DVisco(20, 640, 480)

    train(model, train_loader, learning_rate=0.01, num_epochs=3)
    return


if __name__ == "__main__":
    main()
