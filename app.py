import streamlit as st
from src.document_reader import extract_text
import tempfile

st.title("AI Handwritten Document Intelligence")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    st.image(
        uploaded_file,
        caption="Uploaded Image"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_path = temp_file.name

    text, confidence = extract_text(temp_path)

    st.subheader("Extracted Text")
    st.write(text)

    st.subheader("Confidence")
    st.write(f"{confidence:.2%}")

    st.download_button(
        "Download Text",
        data=text,
        file_name="ocr_output.txt",
        mime="text/plain"
    )