"""
Détection d'image de cartes à jouer
Auteur : Rob MULLA (https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)
Description :
Ce code n'est pas le mien, je n'ai fait qu'une étude sur celui-ci pour comprendre le fonctionnement du machine learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys

import datetime

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

dataset = PlayingCardDataset(data_dir='dataset/cards/train')

print(len(dataset))

image, label = dataset[6000]
print(label)
# image.show()

# Get a dictionary associating target values with folder names
data_dir = 'dataset/cards/train'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = 'dataset/cards/train'
dataset = PlayingCardDataset(data_dir, transform)

image, label = dataset[100]
print(image.shape)

# iterate over dataset
for image, label in dataset:
    break

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    break
print(images.shape,labels.shape)



class SimpleCardClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifer(num_classes=53)
print(str(model)[:500])

example_out = model(images)
print(example_out.shape) # [batch_size, num_classes]

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion(example_out, labels)
print(example_out.shape, labels.shape)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = 'dataset/cards/train/'
valid_folder = 'dataset/cards/valid/'
test_folder = 'dataset/cards/test/'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
val_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Simple training loop
num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"On va utiliser les GPU de la carte graphique : {torch.cuda.is_available()}")
print(torch.cuda.get_device_name(0))

model = SimpleCardClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(datetime.datetime.now())

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
    
print(datetime.datetime.now())

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()

"""
Output:
On va utiliser les GPU de la carte graphique : True
NVIDIA GeForce RTX 4060
2025-10-21 22:38:10.224493
Epoch 1/5 - Train loss: 1.5730812204346951, Validation loss: 0.4310301015961845
Epoch 2/5 - Train loss: 0.5531087991823554, Validation loss: 0.22188419823376637
Epoch 3/5 - Train loss: 0.3369103330036025, Validation loss: 0.2454441288732133
Epoch 4/5 - Train loss: 0.2511437810949614, Validation loss: 0.15516745076989227
Epoch 5/5 - Train loss: 0.17768686839378392, Validation loss: 0.18853449281656517
2025-10-21 22:39:19.655327
"""