import os
from preprocessing.loader import create_dataloaders
from modelling.model import CNN3DVisco
from modelling.train import train
from modelling.test import test
import torch

torch.manual_seed(0)


########## TEMPORARY #############

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def visualize_first_5_frames(data_loader):
    """Visualizes the first 5 frames of the first video in the DataLoader.

    Args:
        data_loader (DataLoader): DataLoader to extract the videos from.
    """
    # Get the first batch from the DataLoader
    data_iter = iter(data_loader)
    data = next(data_iter)

    # Assuming the dataset returns (frames, label) where frames is a tensor
    frames, label = data  # Unpack the batch

    # Get the first video in the batch (assuming batch dimension is first)
    video_frames = frames[0]  # First video in the batch

    # Ensure the video has at least 5 frames (adjust if needed)
    num_frames_to_visualize = min(5, video_frames.shape[1])  # Assuming video shape is [channels, frames, height, width]

    # Create subplots for 5 frames
    fig, axs = plt.subplots(1, num_frames_to_visualize, figsize=(20, 5))  # One row, 5 columns

    # Loop over the first 5 frames of the video
    for idx in range(num_frames_to_visualize):
        # Extract the current frame
        frame = video_frames[:, idx, :, :]  # Assuming [channels, frames, height, width]

        # Convert the frame to a PIL image for visualization
        image = to_pil_image(frame)

        # Display the frame
        axs[idx].imshow(image)
        axs[idx].set_title(f"Frame {idx + 1}")
        axs[idx].axis('off')

    plt.show()

####################################

def main() -> None:
    torch.cuda.empty_cache()
    processed_data_path = os.path.join("data", "processed")
    train_loader, test_loader, valid_loader = create_dataloaders(
        batch_size=16,
        processed_data_path=processed_data_path,
        validation_size=0.15,
        test_size=0.15,
        augment_train_data=True
    )

    visualize_first_5_frames(train_loader)
    visualize_first_5_frames(test_loader)

    #model = CNN3DVisco(55, 199, 196)

    #train(model, train_loader, valid_loader, 0.001, 20)

    # model.load_state_dict(state_dict=torch.load("trained_model.pth", weights_only=True))
    # model.eval()

    # test(model, test_loader)
    return


if __name__ == "__main__":
    main()
