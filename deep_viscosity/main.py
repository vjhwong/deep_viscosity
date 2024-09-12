import os
from loader import create_reg_datasets


def main():
    processed_data_path = os.path.join("data", "processed")
    train_loader, test_loader, valid_loader = create_reg_datasets(
        batch_size=8, processed_data_path=processed_data_path
    )
    for X, y in train_loader:
        print(y)
    return


if __name__ == "__main__":
    main()
