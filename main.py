from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO
from ocr.ocr import OCRModel
from vector_database.vector_database import VectorDatabase
from interface.app import launch_app

from tqdm import tqdm
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    should_launch_app = True

    if should_launch_app:
        launch_app()
    else:
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

        while True:
            search_text = input("Enter search text: ")

            if search_text == "exit":
                break

            search_results = vector_database.search_vector(search_text)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            prompt = (
                f"A user searched for: '{search_text}'\n"
                f"The search returned the following relevant results:\n {search_results[0]['text']}\n"
            )

            messages = [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Respond concisely and in the same language as the user's query. If the user's query is in Hungarian, respond in Hungarian. If the query is in Romanian, respond in Romanian. If the query is in English, respond in English."
                },
                {   "role": "user", 
                    "content": prompt
                }
            ]

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer([text], return_tensors="pt")
            inputs = inputs.to(device)

            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                num_return_sequences=1,
                temperature=0.3,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(response.split("assistant")[-1].strip())