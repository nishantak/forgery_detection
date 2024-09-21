import pandas as pd
from cnn_model import Model
import torch

# Labels: 0 = Authentic, 1 = Forged
data = pd.read_csv(r"model/dataset/train.csv")

model = Model(freeze_layers=6)
model.train(data, num_epochs=16, patience=8, num_folds=9)

torch.save(model.state_dict(), 'model/model.pth')