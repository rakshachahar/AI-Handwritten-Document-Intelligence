import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.cnn_model import EMNISTCNN

transform = transforms.ToTensor()

train_dataset = datasets.EMNIST(
    root="../data",
    split="letters",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.EMNIST(
    root="../data",
    split="letters",
    train=False,
    download=True,
    transform=transform
)

from torch.utils.data import Subset

small_train = Subset(
    train_dataset,
    range(5000)
)

train_loader = DataLoader(
    small_train,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = EMNISTCNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Setup Complete")

num_epochs = 3
for epoch in range(num_epochs):

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/emnist_cnn.pth"
)

print("Model Saved Successfully")