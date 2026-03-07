try:
    import easyocr
except ImportError:
    easyocr = None
import os
from typing import Optional

# Singleton instance per session
_ocr_reader_instance = None

def _get_ocr_reader():
    global _ocr_reader_instance
    if easyocr is None:
        return None
    if _ocr_reader_instance is None:
        print("Loading EasyOCR Reader (CPU)... This may take a moment.")
        try:
            # Initialize reader (will download model on first run if not present)
            _ocr_reader_instance = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR Reader loaded successfully!")
        except Exception as e:
            print(f"Error loading EasyOCR Reader: {e}")
            _ocr_reader_instance = None
    return _ocr_reader_instance

class ImageOCRProcessor:
    """Wrapper around EasyOCR for extracting math problems from images."""

    def process_image(self, image_data) -> Optional[str]:
        """
        Processes image data (bytes, filepath, or numpy array) and returns extracted text.
        """
        try:
            reader = _get_ocr_reader()
            if reader is None:
                return "Error: OCR Reader failed to initialize."
            
            # detail=0 returns only the text list
            results = reader.readtext(image_data, detail=0)
            
            if not results:
                print("OCR: No text found in image.")
                return ""
                
            text = " ".join(results)
            print(f"OCR Extracted: {text[:50]}...")
            return text.strip()
        except Exception as e:
            print(f"OCR Process Error: {e}")
            return f"Error during OCR processing: {str(e)}"
