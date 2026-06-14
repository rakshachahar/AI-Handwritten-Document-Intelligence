import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import datasets, transforms

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

print("Training Samples:", len(train_dataset))
print("Testing Samples:", len(test_dataset))

image, label = train_dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(10):
    image, label = train_dataset[i]

    row = i // 5
    col = i % 5

    axes[row, col].imshow(
        image.squeeze(),
        cmap="gray"
    )

    axes[row, col].set_title(
        f"Label:{label}"
    )

    axes[row, col].axis("off")

plt.tight_layout()
plt.show()

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("DataLoaders Created Successfully")
images, labels = next(iter(train_loader))

print("Batch Image Shape:", images.shape)
print("Batch Label Shape:", labels.shape)