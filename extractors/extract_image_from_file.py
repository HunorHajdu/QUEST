import os
import sys
import pandas as pd
from PIL import Image
import io
from pdf2image import convert_from_path
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------

# Base directory containing all sub-datasets
# Ensure this path is correct. It should contain subfolders like data_EN, data_HU, data_RO
base_datasets_dir = 'datasets'  # Replace with your actual datasets directory if different

# Subdirectories for image extraction
image_output_subdirs = {
    'train': 'train',
    'val': 'val',
    'test': 'test'
}

# Directory for PDF-converted images
pdf_output_subdir = 'pdf_images'

# DPI (resolution) for PDF to image conversion
pdf_dpi = 300

# ------------------------------
# Function Definitions
# ------------------------------

def list_subdatasets(base_dir):
    """
    Lists available sub-datasets in the base directory.

    :param base_dir: Path to the base datasets directory.
    :return: List of sub-dataset names.
    """
    try:
        subdatasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        return subdatasets
    except FileNotFoundError:
        print(f"Error: The directory '{base_dir}' does not exist.")
        sys.exit(1)

def select_subdataset(subdatasets):
    """
    Prompts the user to select a sub-dataset from the available options.

    :param subdatasets: List of available sub-datasets.
    :return: Selected sub-dataset name.
    """
    print("Available Sub-datasets:")
    for idx, sub in enumerate(subdatasets, 1):
        print(f"{idx}. {sub}")
    
    while True:
        try:
            choice = int(input(f"Select a sub-dataset to process (1-{len(subdatasets)}): "))
            if 1 <= choice <= len(subdatasets):
                return subdatasets[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(subdatasets)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def create_output_directories(selected_subdataset):
    """
    Creates necessary output directories for image extraction and PDF conversion.

    :param selected_subdataset: Name of the selected sub-dataset.
    :return: Dictionary of output directories.
    """
    output_dirs = {}
    # Base output directory for the selected sub-dataset
    sub_output_base = os.path.join(base_datasets_dir, selected_subdataset)
    
    # Create train, val, test directories
    for split, subdir in image_output_subdirs.items():
        path = os.path.join(sub_output_base, subdir)
        os.makedirs(path, exist_ok=True)
        output_dirs[split] = path
    
    # Create PDF images directory
    pdf_images_path = os.path.join(sub_output_base, pdf_output_subdir)
    os.makedirs(pdf_images_path, exist_ok=True)
    output_dirs['pdf_images'] = pdf_images_path
    
    return output_dirs

def extract_images_from_parquet(parquet_file, split, output_dir):
    """
    Extracts images from a .parquet file and saves them to the specified directory.

    :param parquet_file: Path to the .parquet file.
    :param split: One of 'train', 'val', 'test'.
    :param output_dir: Directory to save the extracted images.
    """
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error reading {parquet_file}: {e}")
        return

    # Initialize progress bar
    with tqdm(total=len(df), desc=f'Extracting images from {os.path.basename(parquet_file)}', unit='image') as pbar:
        for idx, row in df.iterrows():
            try:
                # Adjust the column access based on your actual data structure
                image_bytes = row['image']['bytes']  # Replace 'image' and 'bytes' if different
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # Ensure image is in RGB format

                # Create a unique filename using 'idx' and row index
                image_filename = os.path.join(output_dir, f"{split}_image_{row['idx']}_{idx}.png")
                image.save(image_filename, format='PNG')
            except Exception as e:
                print(f"\nFailed to process row {idx} in {parquet_file}: {e}")
            finally:
                pbar.update(1)

def convert_pdf_to_images(pdf_file, output_dir, dpi=300):
    """
    Converts each page of a PDF file into separate image files.

    :param pdf_file: Path to the PDF file.
    :param output_dir: Directory to save the converted images.
    :param dpi: Resolution of the output images.
    """
    try:
        pages = convert_from_path(pdf_file, dpi=dpi)
    except Exception as e:
        print(f"Error converting {pdf_file}: {e}")
        return

    pdf_basename = os.path.splitext(os.path.basename(pdf_file))[0]
    
    # Initialize progress bar
    with tqdm(total=len(pages), desc=f'Converting {os.path.basename(pdf_file)}', unit='page') as pbar:
        for page_num, page in enumerate(pages, 1):
            try:
                # Create a unique image filename for each page
                image_filename = os.path.join(output_dir, f"{pdf_basename}_page_{page_num}.png")
                page.save(image_filename, 'PNG')
            except Exception as e:
                print(f"\nFailed to save page {page_num} of {pdf_file}: {e}")
            finally:
                pbar.update(1)

def process_parquet_files(selected_dir, output_dirs):
    """
    Processes all .parquet files in the selected directory.

    :param selected_dir: Path to the selected sub-dataset directory.
    :param output_dirs: Dictionary of output directories.
    """
    parquet_files = [f for f in os.listdir(selected_dir) if f.lower().endswith('.parquet')]
    
    if not parquet_files:
        print("No .parquet files found for image extraction.")
        return
    
    for filename in parquet_files:
        # Determine the split based on filename
        filename_lower = filename.lower()
        if 'train' in filename_lower:
            split = 'train'
        elif 'val' in filename_lower:
            split = 'val'
        elif 'test' in filename_lower:
            split = 'test'
        else:
            print(f"Skipping {filename}: Split not identified in filename.")
            continue

        parquet_path = os.path.join(selected_dir, filename)
        extract_images_from_parquet(parquet_path, split, output_dirs[split])

def process_pdf_files(selected_dir, output_dirs):
    """
    Processes all PDF files in the selected directory.

    :param selected_dir: Path to the selected sub-dataset directory.
    :param output_dirs: Dictionary of output directories.
    """
    pdf_files = [f for f in os.listdir(selected_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found for conversion.")
        return
    
    for filename in pdf_files:
        pdf_path = os.path.join(selected_dir, filename)
        convert_pdf_to_images(pdf_path, output_dirs['pdf_images'], dpi=pdf_dpi)

# ------------------------------
# Main Execution
# ------------------------------

def main():
    print("=== Dataset Image Extraction and PDF Conversion Script ===\n")
    
    # Step 1: List available sub-datasets
    subdatasets = list_subdatasets(base_datasets_dir)
    if not subdatasets:
        print(f"No sub-datasets found in '{base_datasets_dir}'. Ensure the directory contains 'data_EN', 'data_HU', 'data_RO', etc.")
        sys.exit(1)
    
    # Step 2: Prompt user to select a sub-dataset
    selected_subdataset = select_subdataset(subdatasets)
    print(f"\nSelected Sub-dataset: {selected_subdataset}\n")
    
    # Step 3: Create necessary output directories
    output_dirs = create_output_directories(selected_subdataset)
    
    # Path to the selected sub-dataset directory
    selected_dir = os.path.join(base_datasets_dir, selected_subdataset)
    
    # Step 4: Extract images from .parquet files
    print("Starting image extraction from .parquet files...")
    process_parquet_files(selected_dir, output_dirs)
    print("Completed image extraction from .parquet files.\n")
    
    # Step 5: Convert PDF files to images
    print("Starting PDF to image conversion...")
    process_pdf_files(selected_dir, output_dirs)
    print("Completed PDF to image conversion.\n")
    
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
