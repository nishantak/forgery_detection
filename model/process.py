import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from torchvision import transforms


# preprocessing image
def preprocess_image(image_path, img_size=224):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
    resized_image = cv2.resize(edges, (img_size, img_size))
    normalized_image = resized_image / 255.0

    # plt.imshow(normalized_image, cmap='gray')
    # plt.title("Preprocessed Image")
    # plt.show()

    return normalized_image


# pixel analysis 
def texture_analysis(image_path):
    image = cv2.imread(image_path, 0)
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_normalized = (lbp - np.min(lbp)) / (np.max(lbp) - np.min(lbp))

   # plt.imshow(lbp_normalized, cmap='gray')
   # plt.title("Texture Analysis with LBP")
   # plt.show()

    return lbp_normalized


# data augmentation 
augment_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

def augment_image(image):
    augmented_image = augmentat_transforms(image)
    return augmented_image



