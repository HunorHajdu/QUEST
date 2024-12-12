from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO
from ocr.ocr import OCRModel
from vector_database.vector_database import VectorDatabase
from transformers import RagTokenizer, RagSequenceForGeneration
from tqdm import tqdm
import logging
import traceback
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    datasets = []

    # dataset_names = ["EN", "HU", "RO"]
    # dataset_classes = [DataEN, DataHU, DataRO]
    dataset_names = ["HU"]
    dataset_classes = [DataHU]

    dataset_split = "train"
    limit = 1

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

    search_text = "Mi a csak a mentes weboldala?"
    search_results = vector_database.search_vector(search_text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = (
        f"The user searched for: {search_text}\n"
        f"The search returned the following relevant results:\n"
    )
    for idx, result in enumerate(search_results, 1):
        text = result.get('text', 'No text available')
        prompt += f"{idx}. {text}\n\n"
    prompt += "Based on the above results, generate a helpful and informative response."

    print(prompt)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)


    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        num_return_sequences=1,
        temperature=0.7,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)