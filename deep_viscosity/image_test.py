from torchvision import transforms
from PIL import Image

# Open the original image
image_original = Image.open('data/processed/jpg/1.00_1/frame127.jpg')

# Show the undedited image 
image_original.show()

# Define a transform to normalize the image
transform = transforms.Compose([
  transforms.ToTensor(), 
  transforms.Normalize(mean=[0.5], std=[0.5])
  ])

# Apply the transform to the image
image_tensor = transform(image_original)

# Convert tensor back to PIL images for display
image_transformed = transforms.ToPILImage()(image_tensor)

# Display the transformed image
image_transformed.show()