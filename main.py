import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset

# Path to the dataset
DATASET_PATH = "./X-ray"

# Hyperparameters
N_STEPS = 1
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 20 
LEARNING_RATE = 0.001
VAL_SPLIT = 0.14
TEST_SPLIT = 0.16

import numpy as np
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# GPU-specific seeds (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=DATASET_PATH, transform=transform)
total_size = len(dataset)
test_size = int(TEST_SPLIT * total_size)
val_size = int(VAL_SPLIT * total_size)
train_size = total_size - test_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

class TimeSeriesDataset(Dataset):
    def __init__(self, base_ds: Dataset, n_steps: int):
        self.base_ds = base_ds
        self.n_steps = n_steps

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]       # x: [C, H, W]
        # add time dimension and repeat
        x = x.unsqueeze(0)            # → [1, C, H, W]
        x = x.repeat(self.n_steps, 1, 1, 1)  # → [T, C, H, W]
        return x, y                   # returns ([T, C, H, W], label)

# wrap each split
train_ts = TimeSeriesDataset(train_ds, N_STEPS)
val_ts   = TimeSeriesDataset(val_ds,   N_STEPS)
test_ts  = TimeSeriesDataset(test_ds,  N_STEPS)

train_loader = DataLoader(train_ts, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ts,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ts,  batch_size=BATCH_SIZE)

import pytorch_spiking as ps  

class CovidSNN(nn.Module):
    def __init__(self, T: int = 1, num_classes: int = 2, img_size: int = 256):
        super().__init__()
        self.T = T
        
        # 1) spiking‑aware conv + ReLU → spikes
        self.conv1 = nn.Conv2d(1,  8,   kernel_size=3, padding=1)
        self.act1  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True) 

        self.conv2 = nn.Conv2d(8,  64,  kernel_size=3, padding=1)
        self.act2  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True)

        # 2) standard pooling (across spatial dims only)
        self.pool  = nn.MaxPool2d(kernel_size=4, stride=2)

        # 3) another conv + spiking activation
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act3  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True)

        # 4) fully connected spiking layers
        #    after one 4×4 pool stride‑2 on 256→127
        flat_feats = 128 * 127 * 127
        self.fc1   = nn.Linear(flat_feats, 128)
        self.act4  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True)
        
        self.fc2   = nn.Linear(128, 64)
        self.act5  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True)
        
        self.fc3   = nn.Linear(64, 8)
        self.act6  = ps.SpikingActivation(nn.ReLU(), spiking_aware_training=True,
                                          return_sequences=True)
        
        # 5) average spikes over time → remove T dimension
        self.temporal_pool = ps.TemporalAvgPool(dim=1)  

        # 6) final read‑out
        self.fc4   = nn.Linear(8, num_classes)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, T, C, H, W]
        returns logits: [B, num_classes]
        """
        B, T, C, H, W = x_seq.shape

        # — merge batch & time dims for convs —
        x = x_seq.view(B * T, C, H, W)  # → [B*T, C, H, W]
        x = self.conv1(x)               # → [B*T,  8, H, W]
        x = x.view(B, T, -1)            # → [B, T, 8*H*W]
        x = self.act1(x)                # → [B, T, 8*H*W]
        x = x.view(B * T,  8, H, W)     # → back to [B*T, 8, H, W]

        x = self.conv2(x)               # → [B*T, 64, H, W]
        x = x.view(B, T, -1)            # → [B, T, 64*H*W]
        x = self.act2(x)                # → [B, T, 64*H*W]
        x = x.view(B * T, 64, H, W)

        x = self.pool(x)                # → [B*T, 64, 127, 127]

        x = self.conv3(x)               # → [B*T,128,127,127]
        x = x.view(B, T, -1)            # → [B, T, 128*127*127]
        x = self.act3(x)                # → [B, T, 128*127*127]

        # — fully‑connected spiking layers —
        x = self.fc1(x)                 # → [B, T, 128]
        x = self.act4(x)                # → [B, T, 128]
        x = self.fc2(x)                 # → [B, T,  64]
        x = self.act5(x)                # → [B, T,  64]
        x = self.fc3(x)                 # → [B, T,   8]
        x = self.act6(x)                # → [B, T,   8]

        # — collapse time via average pooling —
        x = self.temporal_pool(x)       # → [B, 8]

        # — final classification layer —
        logits = self.fc4(x)            # → [B, num_classes]
        return logits

# Train function
def train(model, loader, optimizer, criterion, device):
    model.train()
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Train for one epoch, return avg loss
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        train_loss = running_loss / n_samples
    return train_loss

# Compute average val loss
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            val_loss = running_loss / n_samples
    return val_loss

# Main loop with loss tracking
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
model = CovidSNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []

print(f"Training on {device}")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    tr_loss = train(model, train_loader, optimizer, criterion, device)
    va_loss = validate(model, val_loader, criterion, device)
    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    print(f"  Train Loss: {tr_loss:.4f} — Val Loss: {va_loss:.4f}")


torch.save(model, "snn.pth")
print("Model saved to snn.pth")