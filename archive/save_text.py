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

with open(
    "outputs/extracted_text.txt",
    "w"
) as file:

    file.write(text)

print("Text Saved Successfully")
print(text)