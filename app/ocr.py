import pytesseract
from PIL import Image


def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.

    :param image_path: Path to the image file.
    :return: Extracted text as a string.
    """
    try:
        # Open an image file
        with Image.open(image_path) as img:
            # Use Tesseract to do OCR on the image
            text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
