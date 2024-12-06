from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

class OCRPostProcessor:
    def __init__(self, model_name='atlijas/byt5-is-ocr-post-processing-old-texts'):
        self.correct_ocr = pipeline('text2text-generation', model=model_name, tokenizer=model_name, num_return_sequences=1)

    def post_process(self, dataset, max_length=150, batch_size=32):
        for corrected in self.correct_ocr(KeyDataset(dataset, 'text'), max_length=max_length, batch_size=batch_size):
            dataset[corrected['key']]['detected_text'] = corrected['generated_text']

        return dataset