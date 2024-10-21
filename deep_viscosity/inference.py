import argparse

import cv2
import torch
import numpy as np

from modelling.model import DeepViscosityModel
from preprocessing.utils.transforms import transform
from preprocessing.utils.cropping import get_window_size


def video_to_tensor(video_path: str, masks: np.ndarray) -> torch.Tensor:
    """Convert a video file to a PyTorch tensor.

    Args:
        video_path (str): Path to the video file.
        masks (np.ndarray): The mask to apply to the frames.

    Returns:
        torch.Tensor: A tensor representing the video.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    cap.release()

    top, left, bottom, right = get_window_size(masks)
    transform_function = transform()
    tensors = []
    for mask, frame in zip(masks, frames):
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
        masked_frame = masked_frame[top:bottom, left:right]
        tensors.append(transform_function(masked_frame).squeeze_(0))
    
    # Stack all frames into a single tensor
    return torch.stack(tensors[45:100], dim=0).unsqueeze_(0)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a 3D CNN model for viscosity prediction."
    )

    parser.add_argument(
        "--model_path", type=str, help="Path to the trained model"
    )

    parser.add_argument(
        "--x_dim", type=int, help="Resolution in x dimension of the input data"
    )

    parser.add_argument(
        "--y_dim", type=int, help="Resolution in y dimension of the input data"
    )

    parser.add_argument(
        "--t_dim", type=int, help="Resolution in t dimension of the input data"
    )

    parser.add_argument(
        "--video_path", type=str, help="Path to the video"
    )

    parser.add_argument(
        "--mask_path", type=str, help="Path to the mask"
    )

    args = parser.parse_args()
    mask = np.load(args.mask_path)

    video_tensor = video_to_tensor(args.video_path, mask).unsqueeze_(0)
    model = DeepViscosityModel(args.t_dim, args.x_dim, args.y_dim)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        output = model(video_tensor)
    print(f"Predicted viscosity: {output.item():.2f}")


if __name__ == "__main__":
    main()