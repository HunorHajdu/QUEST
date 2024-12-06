from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO
from ocr.ocr import OCRModel

from tqdm import tqdm
import numpy as np
import logging
import traceback
    
if __name__ == "__main__":
    datasets = []

    # dataset_names = ["EN", "HU", "RO"]
    # dataset_classes = [DataEN, DataHU, DataRO]
    dataset_names = ["HU"]
    dataset_split = "train"
    limit = 5
    dataset_classes = [DataHU]

    ocr = OCRModel("easyocr")
    
    for name, cls in zip(dataset_names, dataset_classes):
        try:
            logging.info(f"Processing {name} dataset")
            data = cls(limit=limit, split=dataset_split).get_data()
            sum = 0
            if hasattr(data, 'keys'):
                for key in data.keys():
                    sum += len(data[key])
            else:
                sum = len(data)
            logging.info(f"Dataset {name} has {sum} images")
            datasets.append(data)
            
        except Exception as e:
            logging.error(f"Error processing {name} dataset")
            traceback.print_exc()

    for dataset in datasets:
        dataset = dataset.map(ocr.apply_ocr)