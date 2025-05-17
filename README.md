# Idiom Detection Model

This repository contains a neural model for detecting idioms in multilingual text. The model uses a combination of BERT, convolutional layers, BiLSTM, and CRF for sequence labeling.

## Model Architecture

The model architecture consists of:
- Pre-trained transformer backbone (BERT/RoBERTa/XLM-R)
- Sequential dilated convolutions
- Bidirectional cross-attention
- BiLSTM layers
- Multi-head attention
- CRF layer for sequence labeling
- Focal loss combined with CRF loss

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repository
2. Install dependencies from requirements.txt
3. Download the pre-trained model weights from [model link]

## Training

To train the model:

```bash
python train.py \
    --model_name bert-base-multilingual-cased \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --max_length 128 \
    --train_path data/train.csv \
    --val_path data/eval.csv \
    --save_dir saved_models \
    --fp16  # Optional: Enable mixed precision training
```

### Training Arguments

- `--model_name`: Base transformer model to use (default: bert-base-multilingual-cased)
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 128)
- `--train_path`: Path to training data CSV
- `--val_path`: Path to validation data CSV
- `--save_dir`: Directory to save model checkpoints
- `--fp16`: Enable mixed precision training (optional)

## Inference

To run inference:

```bash
python inference.py \
    --test_path path/to/test.csv \
    --output_path path/to/output.csv \
    --model_dir saved_models \
    --model_name model_best.pt \
    --model_type bert-base-multilingual-cased \
    --max_length 128
```

### Inference Arguments

- `--test_path`: Path to test CSV file (required)
- `--output_path`: Path to save predictions CSV (required)
- `--model_dir`: Directory containing saved models (required)
- `--model_name`: Model filename to use (default: bert-base-multilingual-cased_best.pt)
- `--model_type`: Base model type (default: bert-base-multilingual-cased)
- `--max_length`: Maximum sequence length (default: 128)

## Data Format

### Input Format

The input CSV files should contain the following columns:
- `id`: Unique identifier for each example
- `language`: Language code of the text
- `sentence`: Input text to process
- For training and evaluation data only: `indices`: List of word indices marking idiom positions

Example:
```csv
id,language,sentence,indices
1,tr,"Başkasını basamak yapanların düşüşü sert olur.","[1,2]"
2,it,"lo mandi a casa mia?",[-1]
```

### Output Format

The model outputs a CSV file with predictions in the following format:
- `id`: Original example ID
- `language`: Original language code
- `indices`: Predicted word indices for idioms, [-1] if no idiom detected

Example:
```csv
id,language,indices
1,tr,"[3, 4]"
2,it,"[-1]"
```

## Model Weights

- Model weights are available at: https://drive.google.com/file/d/1-03UoI69nBSa8h2RlG4PA7blHG2yew0Y/view?usp=sharing
- This model was trained using XLM-Roberta Large backbone, so the inference code should have model_type set as "FacebookAI/xlm-roberta-large".

## External Resources

The model uses pre-trained models from Hugging Face.
These are downloaded automatically when first running the code.
