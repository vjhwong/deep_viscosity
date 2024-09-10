import os
from dataset import Dataset


def main():
    tensor_data_path = os.path.join("data", "processed", "tensor")

    dataset = Dataset(tensor_data_path, 10)
    train_loader, val_loader, test_loader = dataset.create_dataloaders()

    # Iterate through DataLoader
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print("Features:", features)
        print("Labels:", labels)


if __name__ == "__main__":
    main()
