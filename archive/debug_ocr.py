# notebooks/debug_ocr.py

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

from src.document_reader import extract_text

text = extract_text(
    "data/sample_docs/handwritten_note.jpeg"
)

print(text)