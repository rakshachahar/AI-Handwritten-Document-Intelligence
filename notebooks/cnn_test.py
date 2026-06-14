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
from src.cnn_model import EMNISTCNN

model = EMNISTCNN()

dummy_input = torch.randn(32, 1, 28, 28)

output = model(dummy_input)

print("Input Shape :", dummy_input.shape)
print("Output Shape:", output.shape)