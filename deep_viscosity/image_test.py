from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Original image
image_original = Image.open('data/all_frames/1.0_1/frame0.jpg')
image_original.show()

image_masked = Image.open('data/processed/1.0_1/masked_frame_45.jpg')
image_masked.show()

# Normalized image
transform = transforms.Compose([
  transforms.ToTensor(), 
  transforms.Normalize(mean=[0.5], std=[0.5])
  ])
image_tensor_norm = transform(image_original)
image_normalized = transforms.ToPILImage()(image_tensor_norm)
image_normalized.show()

image_masked_tensor_norm = transform(image_masked)
image_masked_normalized = transforms.ToPILImage()(image_masked_tensor_norm)
image_masked_normalized.show()




# # Equalized image
# image_np = np.array(image_original)
# image_equalized_np = cv2.equalizeHist(image_np)
# image_equalized = Image.fromarray(image_equalized_np)
# image_equalized.show()


# ### JUST FOR FUN; these suck ###

# # Equalized then normalized
# image_tensor_eqnorm = transform(image_equalized)
# image_eqnorm = transforms.ToPILImage()(image_tensor_eqnorm)
# image_eqnorm.show()

# # Normalized then equalized
# image_np_norm = np.array(image_normalized)
# image_normeq_np = cv2.equalizeHist(image_np_norm)
# image_normeq = Image.fromarray(image_normeq_np)
# image_normeq.show()