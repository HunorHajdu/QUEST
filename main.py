from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO
from ocr.ocr import OCRModel
from vector_database.vector_database import VectorDatabase

from tqdm import tqdm
import logging
import traceback
from transformers import pipeline

if __name__ == "__main__":
    datasets = []

    # dataset_names = ["EN", "HU", "RO"]
    # dataset_classes = [DataEN, DataHU, DataRO]
    dataset_names = ["HU"]
    dataset_classes = [DataHU]

    dataset_split = "train"
    limit = 10

    ocr_model = "easyocr"
    
    for name, cls in zip(dataset_names, dataset_classes):
        ocr = OCRModel(ocr_model, language=name)
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

    ocr_applied_datasets = []
    for dataset in datasets:
        ocr_applied_datasets.append(dataset.map(ocr.apply_ocr))

    vector_database = VectorDatabase(documents=ocr_applied_datasets[0]["detected_text"])
    
    for text in tqdm(ocr_applied_datasets[0]["detected_text"]):
        vector_database.add_vectors(text)

    search_text = "Mit ir a dokumentumban?"
    search_results = vector_database.search_vector(search_text)

    pipeline = pipeline("text-generation", model="fQwen/Qwen2.5-1.5B-Instruct", do_sample=True)
    prompt = f"""Answer the question: {search_text}?
    Documents contains: {search_results[0]['text']}"""
    generated_text = pipeline(prompt, max_new_tokens=512, num_return_sequences=1)[0]['generated_text']
    print(generated_text)