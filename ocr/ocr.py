from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from paddleocr import PaddleOCR
import keras_ocr

from ocr.post_process.post_process_ocr import OCRPostProcessor

import logging
import numpy as np

class OCRModel:
    def __init__(self, model, post_processor=None):
        self.post_processor = OCRPostProcessor(post_processor) if post_processor else OCRPostProcessor()
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
        
    def apply_ocr(self, data):
        image_np_array = np.array(data['image'])
        detected_text = self.run_ocr(image_np_array)

        if isinstance(detected_text, list):
            detected_text = '\n'.join([text[1] for text in detected_text])

        if not isinstance(detected_text, str):
            detected_text = str(detected_text)

        detected_lines = detected_text.split('\n')

        corrected_lines = []
        for line in detected_lines:
            corrected_line = self.post_processor.post_process(line)
            corrected_lines.append(corrected_line)

        corrected_text = '\n'.join(corrected_lines)

        data['detected_text'] = detected_text
        data['corrected_text'] = corrected_text
        return data