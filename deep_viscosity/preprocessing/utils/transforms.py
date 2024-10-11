import torch
import random
import torchvision.transforms as transforms
from PIL import Image

class LeftPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        tensor_copy = tensor.clone()
        last_columns = tensor_copy[:, :, -self.padding :]
        remaining_columns = tensor_copy[:, :, : -self.padding]
        return torch.cat([last_columns, remaining_columns], dim=-1)


class RightPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        tensor_copy = tensor.clone()
        first_columns = tensor_copy[:, :, : self.padding]
        remaining_columns = tensor_copy[:, :, self.padding :]
        return torch.cat([remaining_columns, first_columns], dim=-1)


class TopPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        tensor_copy = tensor.clone()
        last_rows = tensor_copy[:, -self.padding :, :]
        remaining_rows = tensor_copy[:, : -self.padding, :]
        return torch.cat([last_rows, remaining_rows], dim=-2)


class BottomPadding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, tensor):
        tensor_copy = tensor.clone()
        first_rows = tensor_copy[:, : self.padding, :]
        remaining_rows = tensor_copy[:, self.padding :, :]
        return torch.cat([remaining_rows, first_rows], dim=-2)

class RandomPadding(object):
    def __init__(self, padding):
        self.padding = padding
        random.seed(42)

    def __call__(self, tensor):
        augmentation_number = random.randint(1, 4)
        
        if augmentation_number == 1:
            padding_fn = LeftPadding(self.padding)
        elif augmentation_number == 2:
            padding_fn = RightPadding(self.padding)
        elif augmentation_number == 3:
            padding_fn = BottomPadding(self.padding)
        else:
            padding_fn = TopPadding(self.padding)

        return padding_fn(tensor)

def transform() -> transforms.Compose:
    """Returns a list of transformations to apply to the images.

    Returns:
        transforms.Compose: List of transformations.
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[1])]
    )

def train_transform() -> transforms.Compose:
    """Returns a list of transformations to apply to the images.

    Returns:
        transforms.Compose: List of transformations.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0], std=[1]),
            RandomPadding(padding=100)
         ]
    )

def main():
    import matplotlib.pyplot as plt

    # Load an image
    image_path = "data/processed/1.0_1/masked_frame_45.jpg"
    image = Image.open(image_path)
    image = image.convert("L")

    transformations = train_transform()
    transformed_image_tensor = transformations(image)
    
    # Convert the tensor back to an image for visualization
    transformed_image = transformed_image_tensor.permute(1, 2, 0).numpy()  # CxHxW -> HxWxC
    transformed_image = (transformed_image * 0.5) + 0.5  # De-normalize the image to [0, 1] range
    
    # Display the transformed image
    plt.imshow(transformed_image)
    plt.axis('off')  # Hide axis
    plt.show()



    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image)

    # Apply LeftPadding transform
    padding = 100  # Example padding value
    left_padding_transform = BottomPadding(padding)
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


if __name__ == "__main__":
    main()
