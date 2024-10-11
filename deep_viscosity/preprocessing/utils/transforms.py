import torch
import torchvision.transforms as transforms
from PIL import Image


class LeftPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        last_columns = tensor[:, :, -self.padding :]
        remaining_columns = tensor[:, :, : -self.padding]
        return torch.cat([last_columns, remaining_columns], dim=-1)


class RightPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        first_columns = tensor[:, :, : self.padding]
        remaining_columns = tensor[:, :, self.padding :]
        return torch.cat([remaining_columns, first_columns], dim=-1)


class TopPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        last_rows = tensor[:, -self.padding :, :]
        remaining_rows = tensor[:, : -self.padding, :]
        return torch.cat([last_rows, remaining_rows], dim=-2)


class BottomPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        first_rows = tensor[:, : self.padding, :]
        remaining_rows = tensor[:, self.padding :, :]
        return torch.cat([remaining_rows, first_rows], dim=-2)


def transform() -> transforms.Compose:
    """Returns a list of transformations to apply to the images.

    Returns:
        transforms.Compose: List of transformations.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])]
    )


def main():
    import matplotlib.pyplot as plt

    # Load an image
    image_path = "path_to_your_image.jpg"
    image = Image.open(image_path)

    # Convert image to tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image)

    # Apply LeftPadding transform
    padding = 10  # Example padding value
    left_padding_transform = LeftPadding(padding)
    transformed_tensor = left_padding_transform(image_tensor)

    # Convert tensor back to image for display
    transform_to_pil = transforms.ToPILImage()
    transformed_image = transform_to_pil(transformed_tensor)

    # Display the image
    plt.imshow(transformed_image)
    plt.axis("off")
    plt.show()
    pass


if __name__ == "__main__":
    main()
