from input_processing.image_ocr import ImageOCRProcessor
import os
import requests

def test_ocr():
    print("Testing OCR Processor...")
    ocr = ImageOCRProcessor()
    
    # Download a sample math image for testing if possible, 
    # or just check if it initializes correctly.
    print("OCR Initialized. Attempting to process a dummy image (should fail gracefully or wait for model download).")
    
    # Let's try to list files in the current dir to see if there are any images we can use
    files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if files:
        print(f"Found image for testing: {files[0]}")
        with open(files[0], 'rb') as f:
            data = f.read()
        res = ocr.process_image(data)
        print(f"OCR Result: {res}")
    else:
        print("No local image found for testing.")
        # Just testing initialization
        reader = ocr.process_image(b"fake data")
        print(f"OCR process call completed (Expected error: {reader})")

if __name__ == "__main__":
    test_ocr()
