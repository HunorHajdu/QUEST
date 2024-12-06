from datasets import load_dataset
from local_datasets.data_cleaner.data_cleaner import DataCheckers

class DataHU:
    def __init__(self, limit=None, split=None):
        self.data = load_dataset("nhiremath/HungarianDocQA_IT_SyntheticQA")
        self.data = self.data.remove_columns(["conversations", "markdown"])
        if limit is not None and split is not None:
            self.data = self.data[split].select(range(limit))


    def get_data(self):
        # Data check commented out due to memory issues
        # self.data = DataCheckers(self.data).remove_duplicates()
        return self.data