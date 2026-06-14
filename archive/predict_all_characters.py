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

transform = transforms.ToTensor()

output_folder = "outputs"

files = sorted(
    [
        f for f in os.listdir(output_folder)
        if f.endswith(".png")
    ]
)

for file in files:

    image_path = os.path.join(
        output_folder,
        file
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (28, 28)
    )

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        predicted = torch.argmax(
            output,
            dim=1
        ).item()

    letter = chr(predicted + 64)

    print(
        f"{file} --> {letter}"
    )