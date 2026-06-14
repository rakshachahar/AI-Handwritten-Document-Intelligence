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
import cv2
from torchvision import transforms

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

# Load one cropped image
image = cv2.imread(
    "outputs/char_4.png",
    cv2.IMREAD_GRAYSCALE
)

# Resize to EMNIST size
image = cv2.resize(
    image,
    (28, 28)
)

transform = transforms.ToTensor()

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    predicted = torch.argmax(
        output,
        dim=1
    ).item()

letter = chr(predicted + 64)

print("Predicted Letter:", letter)