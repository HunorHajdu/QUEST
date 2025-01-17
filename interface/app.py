from ocr.ocr import OCRModel
from vector_database.vector_database import VectorDatabase

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tempfile
import os

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'files_processed' not in st.session_state:
        st.session_state.files_processed = False
    if 'vector_database' not in st.session_state:
        st.session_state.vector_database = VectorDatabase() 

def display_chat_message(role, content):
    with st.chat_message(role):
        st.write(content)

def launch_app():
    st.set_page_config(page_title="Q.U.E.S.T. – Quick Understanding and Extraction of Structured Text", layout="wide")
    initialize_session_state()
    
    st.title("Q.U.E.S.T. – Quick Understanding and Extraction of Structured Text")
    
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )

        st.header("Choose an OCR Model")
        ocr_model = st.selectbox(
                "Choose an OCR model",
                ("easyocr", 
                 "trocr", 
                 "paddleocr", 
                 "kerasocr"),
            )
        st.header("Choose a Language")
        language = st.selectbox(
                "Choose a language",
                ("HU", 
                 "EN",
                 "RO"),
            )

        if uploaded_files and not st.session_state.files_processed:
            with st.spinner("Processing documents..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_files = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        pdf_files.append(file_path)
                    
                    ocr = OCRModel(ocr_model, language=language)

                    ocr_applied_texts = []
                    for pdf_file in pdf_files:
                        ocr_applied_texts.append(ocr.single_file_ocr(pdf_file))

                    for text in ocr_applied_texts:
                        st.session_state.vector_database.add_vectors(text['detected_text'])
                        st.session_state.vector_database.add_text(text['detected_text'])

                    st.session_state.files_processed = True
                    st.success(f"✅ {len(uploaded_files)} files processed!")
        
        st.header("Choose a Model")
        model_name = st.selectbox(
            "Choose a model",
            ("HuggingFaceTB/SmolLM2-135M-Instruct", 
             "HuggingFaceTB/SmolLM2-1.7B-Instruct", 
             "Qwen/Qwen2.5-1.5B-Instruct", 
             "Qwen/Qwen2.5-3B-Instruct",
             "Qwen/Qwen2.5-7B-Instruct", 
             "microsoft/phi-4"),
        )
        
    if st.session_state.files_processed and uploaded_files:
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        if prompt := st.chat_input("Ask a question about your documents"):
            display_chat_message("user", prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    search_results = st.session_state.vector_database.search_vector(prompt)

                    system_prompt = (
                        f"Context: {search_results[0]['text']}\n"
                        f"Question: '{prompt}'\n"
                    )

                    messages = [
                        {
                            "role": "system", 
                            "content": """You are a highly skilled and adaptive assistant, capable of understanding and responding to a wide range of queries. Respond to the user's requests with clear, concise, and relevant answers, maintaining a professional and friendly tone.

                            Language: Automatically detect the language of the user's query and respond in that language, whether it's Hungarian, Romanian, English, or another language. If the query is in:

                            -Hungarian, respond fluently in Hungarian.
                            -Romanian, respond fluently in Romanian.
                            -English, respond fluently in English.
                            Contextual Understanding:

                            -If the user refers to specific documents, articles, or any previous content, use that information to craft relevant and accurate responses.
                            -Provide answers based on factual accuracy, especially when explaining or summarizing information.
                            -Always aim for clarity, avoiding jargon, and ensuring that the response can be easily understood."""
                        },
                        {   "role": "user", 
                            "content": system_prompt
                        }
                    ]

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

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
                        max_new_tokens=512,
                        num_return_sequences=1,
                        temperature=0.3,
                    )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    st.write(response.split("assistant")[-1].strip())
                    st.session_state.messages.append({"role": "assistant", "content": response.split("assistant")[-1].strip()})
    else:     
        if not st.session_state.messages:
            display_chat_message("assistant", "Hello! I'm your PDF assistant. Please upload some documents to get started.")
        if not uploaded_files:
            st.session_state.messages = []
        st.session_state.files_processed = False

    if st.session_state.messages:
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
