# Q.U.E.S.T. – Quick Understanding and Extraction of Structured Text

Q.U.E.S.T. is a comprehensive tool designed to facilitate the extraction and understanding of structured text from PDF documents. It leverages OCR (Optical Character Recognition) models and vector databases to process, store, and search text data efficiently.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- **OCR Processing**: Supports multiple OCR models including EasyOCR, TrOCR, PaddleOCR, and KerasOCR.
- **Language Support**: Handles English, Hungarian, and Romanian languages.
- **Vector Database**: Efficiently stores and searches text data using vector embeddings.
- **Streamlit Interface**: User-friendly web interface for uploading documents and querying the processed text.

## Installation

1. Clone the repository:

2. Create and activate a virtual environment, install dependencies and run the application:
    ```sh
    make all
    ```

## Usage

1. Launch the Streamlit application:
    ```sh
    make run
    ```

2. Upload PDF documents through the web interface.

3. Choose an OCR model and process the documents.

4. Query the processed text using the chat interface.

## Project Structure

- **interface/**: Contains the Streamlit application.
- **local_datasets/**: Scripts for handling datasets in different languages.
- **ocr/**: OCR model implementations and post-processing.
- **vector_database/**: Vector database for storing and searching text data.
- **main.py**: Entry point for the application.
