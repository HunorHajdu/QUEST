from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from paddleocr import PaddleOCR
import keras_ocr

import logging

class OCRModel:
    def __init__(self, model):
        if model is None or model.lower() == "easyocr":
            logging.warning("No OCR model provided, using EasyOCR as default")
            self.model = easyocr.Reader(['en', 'ro', 'hu'])
        elif model.lower() == "trocr":
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        elif model.lower() == "paddleocr":
            self.model = PaddleOCR()
        elif model.lower() == "kerasocr":
            self.model = keras_ocr.pipeline.Pipeline(use_angle_cls=True)

    def run_ocr(self, image):
        if isinstance(self.model, easyocr.Reader):
            return self.model.readtext(image)
        elif isinstance(self.model, VisionEncoderDecoderModel):
            return self.processor(image)
        elif isinstance(self.model, PaddleOCR):
            return self.model.ocr(image, cls=False)
        elif isinstance(self.model, keras_ocr.pipeline.Pipeline):
            return self.model.recognize([image])
        else:
            logging.error("Invalid OCR model provided!")
            return None
