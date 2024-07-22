from PIL import Image, ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import fitz  # PyMuPDF
import docx

# Load the OCR model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def preprocess_image(image, contrast_factor=2, median_filter_size=3, binarization_threshold=128):
    """
    Preprocess the image to improve OCR accuracy.
    """
    gray = image.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(contrast_factor)
    denoised = enhanced.filter(ImageFilter.MedianFilter(size=median_filter_size))
    binary = denoised.point(lambda x: 0 if x < binarization_threshold else 255, '1')
    rgb_image = binary.convert("RGB")
    return rgb_image

def segment_image(image, segment_height=400):
    """
    Segments an image into smaller pieces for easier OCR processing.
    """
    width, height = image.size
    segments = []
    for i in range(0, height, segment_height):
        box = (0, i, width, min(i + segment_height, height))
        segment = image.crop(box)
        segments.append(segment)
    return segments

def extract_text_from_image(image_path):
    """
    Extract text from an image file using OCR.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = preprocess_image(image)
        segments = segment_image(preprocessed_image)
        full_text = ""
        for segment in segments:
            pixel_values = trocr_processor(segment, return_tensors="pt").pixel_values
            generated_ids = trocr_model.generate(pixel_values)
            generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            full_text += generated_text + "\n"
        return full_text if full_text.strip() else "No text extracted from the image"
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file, processing each page one at a time.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            text += f"Page {page_num + 1}:\n{page_text}\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """
    Extract text from a DOCX file.
    """
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

def extract_text_from_file(file_path, file_extension):
    """
    Determine the appropriate text extraction function based on file extension.
    """
    if file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        return extract_text_from_image(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format for text extraction.")
