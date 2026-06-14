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
from torchvision import datasets, transforms
from src.cnn_model import EMNISTCNN

# Load model
model = EMNISTCNN()

model.load_state_dict(
    torch.load(
        "models/emnist_cnn.pth",
        map_location="cpu"
    )
)

model.eval()

# Load test dataset
transform = transforms.ToTensor()

test_dataset = datasets.EMNIST(
    root="data",
    split="letters",
    train=False,
    download=True,
    transform=transform
)

# Get one sample
image, label = test_dataset[0]

# Add batch dimension
image = image.unsqueeze(0)

# Prediction
with torch.no_grad():

    output = model(image)

    predicted = torch.argmax(
        output,
        dim=1
    ).item()

print("Actual Label    :", label)
print("Predicted Label :", predicted)
letter = chr(predicted + 64)

actual_letter = chr(label + 64)

print("Actual Letter    :", actual_letter)
print("Predicted Letter :", letter)