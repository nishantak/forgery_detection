import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


# model loader
def load_model():
    model = torch.load('model.pth')
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
        image = cv2.imread(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label


# data loaders
def get_data_loader(image_paths, labels):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = DocumentDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    return loader
