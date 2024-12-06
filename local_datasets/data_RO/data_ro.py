import os
from pdf2image import convert_from_path
from datasets import Dataset
from local_datasets.data_cleaner.data_cleaner import DataCheckers

class DataRO:
    def __init__(self):
        self.pdf_directory = "./datasets/data_RO/"
        self.images = []

    def convert_pdfs(self):
        for pdf_file in os.listdir(self.pdf_directory):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                pages = convert_from_path(pdf_path)
                
                for page in pages:
                    self.images.append(page)

    def create_dataset(self):
        self.convert_pdfs()
        return Dataset.from_list(self.images)
    
    def get_data(self):
        data = self.create_dataset()
        data = DataCheckers(data).remove_duplicates()
        
        return data