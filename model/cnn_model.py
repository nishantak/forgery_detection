import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from process import augment_image, texture_analysis, preprocess_image
from sklearn.model_selection import KFold
import copy
from utils import get_data_loader

class Model:
    def __init__(self, freeze_layers=6):

        self.model = models.resnet18(weights='DEFAULT')
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

        frozen_count = 0
        for _, child in self.model.named_children():
            if frozen_count < freeze_layers:
                if isinstance(child, nn.BatchNorm2d):
                    continue
                for param in child.parameters():
                    param.requires_grad = False
                frozen_count += 1
            else: break 

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)


    # K-fold validation because small dataset
    def train(self, dataset, num_epochs=16, patience=8, num_folds=10):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold + 1}/{num_folds}')

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = get_data_loader(train_subset.dataset.image_path[train_idx], train_subset.dataset.label[train_idx], batch_size=4)
            val_loader = get_data_loader(val_subset.dataset.image_path[val_idx], val_subset.dataset.label[val_idx], batch_size=2)

            self.train_single_fold(train_loader, val_loader, num_epochs, patience)


    def train_single_fold(self, train_dataloader, val_dataloader, num_epochs=16, patience=8):
        self.model.train()
        best_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(num_epochs):
            running_loss = 0.0
            # Training phase
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                combined_features = []
                for img in inputs:
                    augmented_img = augment_image(img)

                    img_np = augmented_img.permute(1, 2, 0).cpu().numpy()
                    texture_features_np = texture_analysis(img_np)
                    texture_features_tensor = torch.tensor(texture_features_np, dtype=torch.float32).unsqueeze(0).to(self.device)

                    combined_img_tensor = torch.cat((augmented_img, texture_features_tensor), dim=1)
                    combined_features.append(combined_img_tensor)
                combined_features = torch.stack(combined_features).to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(combined_features)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_dataloader:
                    val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)

                    combined_features = []
                    for img in val_inputs:
                        texture_features_tensor = torch.tensor(texture_analysis(img.permute(1, 2, 0).cpu().numpy()), dtype=torch.float32).unsqueeze(0).to(self.device)
                        combined_img_tensor = torch.cat((augmented_img, texture_features_tensor), dim=1)
                        combined_features.append(combined_img_tensor)
                    combined_features = torch.stack(combined_features).to(self.device)

                    val_outputs = self.model(combined_features)
                    loss = self.criterion(val_outputs, val_labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            avg_val_loss = val_loss / len(val_dataloader)
            accuracy = 100 * correct / total

            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping")
                break

        self.model.load_state_dict(best_model_wts)

