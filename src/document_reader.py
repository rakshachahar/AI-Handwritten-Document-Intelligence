import easyocr

reader = easyocr.Reader(['en'])

def extract_text(image_path):

    result = reader.readtext(image_path)

    text = ""
    confidence = []

    for item in result:

        text += item[1] + " "
        confidence.append(item[2])

    avg_confidence = sum(confidence) / len(confidence)

    return text.strip(), avg_confidence