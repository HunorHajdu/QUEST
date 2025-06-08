from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from paddleocr import PaddleOCR
import keras_ocr
from PIL import Image
from ocr.post_process.post_process_ocr import OCRPostProcessor

import logging
import numpy as np
from pdf2image import convert_from_path


from quest_ocr.infer import load_ocr_model, predict_text


class OCRModel:
    def __init__(self, model, language="EN"):
        if language not in ["EN", "RO", "HU"]:
            logging.error("Invalid language provided, using EN as default")
            language = "EN"
        elif language == "EN":
            post_processor = "oliverguhr/spelling-correction-english-base"
        elif language == "HU":
            post_processor = "NYTK/ocr-cleaning-mt5-base-hungarian"
        elif language == "RO":
            post_processor = "iliemihai/mt5-base-romanian-diacritics"

        self.post_processor = OCRPostProcessor(post_processor)

        if model is None or model.lower() == "easyocr":
            logging.warning("No OCR model provided, using EasyOCR as default")
            self.model = easyocr.Reader([language.lower()])
        elif model.lower() == "trocr":
            self.processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten"
            )
        elif model.lower() == "paddleocr":
            self.model = PaddleOCR()
        elif model.lower() == "kerasocr":
            self.model = keras_ocr.pipeline.Pipeline()
        elif model.lower() == "quest-ocr":
            self.model = model.lower()
        elif model.lower() == "quest-ocr-hierarchical":
            self.model = model.lower()

    def run_ocr(self, image):
        if isinstance(self.model, easyocr.Reader):
            return self.model.readtext(image)
        elif isinstance(self.model, VisionEncoderDecoderModel):
            return self.processor(image)
        elif isinstance(self.model, PaddleOCR):
            return self.model.ocr(image, cls=False)
        elif isinstance(self.model, keras_ocr.pipeline.Pipeline):
            return self.model.recognize([image])
        elif self.model.lower() == "quest-ocr":
            simple_model_path = "models/quest-simple.pth"
            image = self.numpy_to_pil(image)
            return predict_text(simple_model_path, image)
        elif self.model.lower() == "quest-ocr-hierarchical":
            hierarchical_model_path = "models/quest-hierarchical.pth"
            image = self.numpy_to_pil(image)
            return predict_text(hierarchical_model_path, image)
        else:
            logging.error("Invalid OCR model provided!")
            return None

    def numpy_to_pil(self, np_array):
        """
        Convert numpy array to PIL Image

        Args:
            np_array: NumPy array (can be 2D grayscale or 3D RGB/RGBA)

        Returns:
            PIL Image
        """
        # Handle different array shapes
        if len(np_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(np_array, mode="L")
        elif len(np_array.shape) == 3:
            if np_array.shape[2] == 3:
                # RGB image
                image = Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                # RGBA image
                image = Image.fromarray(np_array, mode="RGBA")
            else:
                raise ValueError(f"Unsupported array shape: {np_array.shape}")
        else:
            raise ValueError(f"Array must be 2D or 3D, got shape: {np_array.shape}")

        return image

    def apply_ocr(self, data):
        image_np_array = np.array(data["image"])
        detected_text = self.run_ocr(image_np_array)

        if isinstance(detected_text, list):
            detected_text = "\n".join([text[1] for text in detected_text])
        if not isinstance(detected_text, str):
            detected_text = str(detected_text)

        corrected_lines = []
        for line in detected_text.splitlines():
            corrected_line = self.post_processor.post_process(line)
            corrected_lines.append(corrected_line)

        corrected_text = "\n".join(corrected_lines)

        return {"detected_text": detected_text, "corrected_text": corrected_text}

    def single_file_ocr(self, pdf_path):
        pages = convert_from_path(pdf_path)
        detected_text = []

        for page in pages:
            image_np_array = np.array(page)
            page_text = self.run_ocr(image_np_array)
            if isinstance(page_text, list):
                page_text = "\n".join([text[1] for text in page_text])
            if not isinstance(page_text, str):
                page_text = str(page_text)

            detected_text.append(page_text)

        # corrected_lines = []
        # for line in detected_text:
        #     corrected_line = self.post_processor.post_process(line)
        #     corrected_lines.append(corrected_line)

        # corrected_text = '\n'.join(corrected_lines)

        print(f"Detected text from {pdf_path}: {detected_text}")

        return {
            "detected_text": detected_text
            # 'corrected_text': corrected_text
        }
