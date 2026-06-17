# MTr-VQA Model
---

## Table of Contents
MTr-VQA: Multi-transformer visual question answering with improved generalization and interpretability
1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Usage Guide](#usage-guide)
   - [Dataset Preparation](#dataset-preparation)
   - [Training the Model](#training-the-model)
   - [Testing the Model](#testing-the-model)

---

## System Requirements
- **Python**: 3.10
- **Dependencies**: Listed in `requirements.txt`

---

## Environment Setup

1. **Clone the Repository**  
   Open your terminal and run:
   ```bash
   git clone https://github.com/Hovohoangduy/MTr-VQA.git
   cd MTr-VQA
   ```

2. **Install Dependencies**  
   Use `pip` to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### Dataset Preparation
1. **Download Dataset**  
   Download the ViTextVQA dataset

2. **Convert JSON to CSV**  
   Use the following command to convert JSON data into CSV format:
   ```bash
   python -m utils.json_to_csv \
   --json_folder_path "Path to folder containing JSON files" \
   --csv_folder_path "Path to folder where CSV files will be saved"
   ```

---

### Training the Model
To train the MTr-VQA model, run the following command with the appropriate arguments:

```bash
python -m train \
 --batch_size 4 \
 --epochs 10 \
 --img_path "Path to image folder" \
 --model_path "Path to save trained model" \
 --train_csv_path "Path to training CSV file" \
 --test_csv_path "Path to testing CSV file" \
 --dev_csv_path "Path to development CSV file"
```

**Arguments**:  
- `--batch_size`: Size of each training batch (default: 4).  
- `--epochs`: Number of training epochs (default: 10).  
- `--img_path`: Directory containing input images.  
- `--model_path`: Directory to save the trained model.  
- `--train_csv_path`: Path to the training dataset CSV file.  
- `--test_csv_path`: Path to the testing dataset CSV file.  
- `--dev_csv_path`: Path to the development dataset CSV file.

---

### Testing the Model
To test the model's performance, run the following command:

```bash
python -m test \
 --img_path "Path to image folder" \
 --model_path "Path to saved model" \
 --train_csv_path "Path to training CSV file" \
 --test_csv_path "Path to testing CSV file" \
 --dev_csv_path "Path to development CSV file"
```

**Arguments**:  
- `--img_path`: Directory containing input images.  
- `--model_path`: Path to the trained model file.  
- `--train_csv_path`: Path to the training dataset CSV file.  
- `--test_csv_path`: Path to the testing dataset CSV file.  
- `--dev_csv_path`: Path to the development dataset CSV file.

---
