import os
from pdf2image import convert_from_path
from datasets import Dataset
from tqdm import tqdm

class DataRO:
    def __init__(self):
        self.pdf_directory = "./local_datasets/data_RO/"
        self.images = []

    def create_dataset(self):
        for pdf_file in tqdm(os.listdir(self.pdf_directory), desc="Checking PDFs for DataRO"):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                pages = convert_from_path(pdf_path)
                
                for page in pages:
                    self.images.append({"image": page})
        return Dataset.from_list(self.images)
    
    def get_data(self):
        data = self.create_dataset()
        
        return data