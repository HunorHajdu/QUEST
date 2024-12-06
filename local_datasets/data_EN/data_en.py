import os
from datasets import Dataset, load_dataset, concatenate_datasets
from pdf2image import convert_from_path
from local_datasets.data_cleaner.data_cleaner import DataCheckers
from tqdm import tqdm

class DataEN:
    def __init__(self, limit=None, split=None):
        self.hf_data = load_dataset("aharley/rvl_cdip")
        self.pdf_directory = "./local_datasets/data_EN/"
        self.should_check_pdfs = limit is None and split is None
        
        if self.should_check_pdfs:
            local_images = []
            for pdf in tqdm(os.listdir(self.pdf_directory), desc="Checking PDFs for DataEN"):
                if pdf.endswith('.pdf'):
                    pdf_path = os.path.join(self.pdf_directory, pdf)
                    pages = convert_from_path(pdf_path)
                    local_images.extend(pages)
        
            local_images_dict = [{"image": img} for img in local_images]
            local_dataset = Dataset.from_list(local_images_dict)
        
            self.data = concatenate_datasets([self.hf_data['train'], local_dataset])
        else:
            self.data = self.hf_data['train'].select(range(limit))
    def get_data(self):
        # Data check commented out due to memory issues
        # self.data = DataCheckers(self.data).remove_duplicates()

        return self.data