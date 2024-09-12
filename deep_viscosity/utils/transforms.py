import torchvision.transforms as transforms


def transform() -> transforms.Compose:
    """Returns a list of transformations to apply to the images.

    Returns:
        transforms.Compose: List of transformations.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
