from datasets import load_dataset
from local_datasets.data_cleaner.data_cleaner import DataCheckers

class DataHU:
    def __init__(self):
        self.data = load_dataset("nhiremath/HungarianDocQA_IT_SyntheticQA")

    def get_data(self):
        # self.data = DataCheckers(self.data).remove_duplicates()
        
        return self.data