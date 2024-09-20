import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from process import preprocess_image

# model loader
def load_model():
    from cnn_model import Model

    model = Model()
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    return model


# document loader
class DocumentDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        label = self.labels[idx]

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        
        return image, label


# data loaders
def get_data_loader(image_paths, labels, batch_size=16):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DocumentDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader
