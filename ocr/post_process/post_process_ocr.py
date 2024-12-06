from transformers import pipeline, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset

class OCRPostProcessor:
    def __init__(self, model_name='google/mt5-small'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.pipeline = pipeline('text2text-generation', model=model_name, tokenizer=tokenizer)

    def post_process(self, text):
        return self.pipeline(text)