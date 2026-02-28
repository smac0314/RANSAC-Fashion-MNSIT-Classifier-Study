import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split


print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2859], [0.3530])
])
FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
FASHION_train = Subset(FASHION_trainval, range(50000))
FASHION_val = Subset(FASHION_trainval, range(50000,60000))
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")

trainloader = DataLoader(FASHION_train, batch_size=64, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size=64, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=64, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = 32

        self.classifier = nn.Sequential(
            nn.Conv2d(1, kernels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(kernels),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kernels, kernels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(kernels*2),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kernels*2, kernels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(kernels*4),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(kernels*4*3*3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" 
model = Network().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epoch = 14
train_losses = []
val_losses = []

def compute_val_loss(model, loader):
    losses = []
    with torch.no_grad():
        for batch, label in loader:
            batch, label = batch.to(device), label.to(device)
            losses.append(criterion(model(batch), label).item())
    return np.mean(losses)

def train(model, loader, num_epoch = 14):
    print("Start training...")
    model.train()
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, label)
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_train_loss = np.mean(running_loss)
        train_losses.append(epoch_train_loss)
        epoch_val_loss = compute_val_loss(model, valloader)
        val_losses.append(epoch_val_loss)
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss)))
    print("Done!")

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker='o', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses,   marker='s', linewidth=2, label='Validation Loss')
    plt.title('Training Loss vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./part2_2.png', dpi=150)
    plt.show()
    
train(model, trainloader, num_epoch)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)
plot_losses(train_losses, val_losses)
