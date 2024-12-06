import logging
from cleanvision.imagelab import Imagelab
from datasets import load_dataset


class DataCheckers:
    def __init__(self, dataset):
        self.data = dataset
        self.imagelab = None

    def check_data(self):
        self.imagelab = Imagelab(hf_dataset=self.data, image_key="image")
        self.imagelab.find_issues()
        self.imagelab.report()
    
    def remove_duplicates(self):
        if self.imagelab is None:
            logging.warning("Running check_data()!")
            self.check_data()
        
        duplicates = self.imagelab.info['exact_duplicates']['sets']
        indices_to_remove = set()
    
        for duplicate_set in duplicates:
            indices_to_remove.update(list(duplicate_set)[1:])
        
        self.data = self.data.filter(
            lambda _, idx: idx not in indices_to_remove, 
            with_indices=True
        )
        
        logging.info(f"Removed {len(indices_to_remove)} duplicate images")
        
        return self.data