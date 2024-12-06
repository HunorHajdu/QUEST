import os
from pdf2image import convert_from_path
from datasets import Dataset
from tqdm import tqdm

class DataRO:
    def __init__(self, limit=None, split=None):
        self.pdf_directory = "./local_datasets/data_RO/"
        self.images = []
        self.limit = limit
        self.split = split

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

        if self.limit is not None and self.split is not None:
            self.data = self.data[self.split].select(range(self.limit))

        return data