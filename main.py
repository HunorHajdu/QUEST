from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO
from ocr.ocr import OCRModel

from tqdm import tqdm
import traceback

if __name__ == "__main__":
    datasets = []
    dataset_names = ["EN", "HU", "RO"]
    dataset_classes = [DataEN, DataHU, DataRO]
    ocr = OCRModel("easyocr")

    
    for name, cls in zip(dataset_names, dataset_classes):
        try:
            print(f"Processing {name} dataset...")
            data = cls().get_data()
            sum = 0
            if hasattr(data, 'keys'):
                for key in data.keys():
                    sum += len(data[key])
            else:
                sum = len(data)
            print(f"Total number of images in {name} dataset: {sum}")
            datasets.append(data)
            
        except Exception as e:
            print(f"Error processing {name} dataset: {e}")
            traceback.print_exc()

    for dataset in datasets:
        for i in tqdm(range(len(dataset)), desc="Running OCR"):
            dataset[i]['detected_text'] = ocr.get_text(dataset[i]['image'])