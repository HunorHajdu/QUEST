import os
from datasets import Dataset, load_dataset, concatenate_datasets
from pdf2image import convert_from_path
from data_cleaner.data_cleaner import DataCheckers

class DataEN:
   def __init__(self):
       self.hf_data = load_dataset("aharley/rvl_cdip")
       self.pdf_directory = "./datasets/data_EN/"
       
       local_images = []
       for pdf in os.listdir(self.pdf_directory):
           pdf_path = os.path.join(self.pdf_directory, pdf)
           pages = convert_from_path(pdf_path)
           local_images.extend(pages)
       
       local_dataset = Dataset.from_list(local_images)
       
       self.data = concatenate_datasets([self.hf_data, local_dataset])

   def get_data(self):
       self.data = DataCheckers(self.data).remove_duplicates()

       return self.data