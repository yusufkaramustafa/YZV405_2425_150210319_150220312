import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class IdiomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        words = sentence.split()
        word_tokens = []
        word_label_ids = []
        
        # Add CLS token
        word_tokens.append(self.tokenizer.cls_token)
        word_label_ids.append(0)
        
        # Process words and their labels
        for word, word_label in zip(words, label):
            tokens = self.tokenizer.tokenize(word)
            if not tokens:
                continue
                
            word_tokens.extend(tokens)
            word_label_ids.append(word_label)
            
            # Handle subwords
            if len(tokens) > 1:
                if word_label == 1:
                    word_label_ids.extend([2] * (len(tokens) - 1))
                else:
                    word_label_ids.extend([word_label] * (len(tokens) - 1))
        
        # Add SEP token
        word_tokens.append(self.tokenizer.sep_token)
        word_label_ids.append(0)
        
        # Truncate if needed
        if len(word_tokens) > self.max_length:
            word_tokens = word_tokens[:self.max_length]
            word_label_ids = word_label_ids[:self.max_length]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            word_label_ids.extend([0] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(word_label_ids)
        }

def preprocess_data(df, tokenizer):
    inputs, labels = [], []
    label_counts = {0: 0, 1: 0, 2: 0}

    print("\nProcessing examples...")
    for idx, row in enumerate(df.iterrows()):
        _, row = row
        sentence = row["sentence"]
        expression = row["expression"]
        category = row["category"]
        
        try:
            idiom_indices = eval(row["indices"])
            tokenized_words = eval(row["tokenized_sentence"])
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(f"Row data: {row}")
            continue
        
        # Debug info for first 5 examples
        if idx < 5:
            print(f"\nExample {idx}:")
            print(f"Sentence: {sentence}")
            print(f"Expression: {expression}")
            print(f"Category: {category}")
            print(f"Indices: {idiom_indices}")
            print(f"Tokenized words: {tokenized_words}")
        
        bio_tags = [0] * len(tokenized_words)
        
        # Process idiomatic expressions
        if category == "idiomatic" and idiom_indices != [-1]:
            valid_indices = [idx for idx in idiom_indices if idx < len(tokenized_words)]
            if len(valid_indices) != len(idiom_indices):
                print(f"\nWarning: Invalid indices in idiomatic expression:")
                print(f"Sentence: {sentence}")
                print(f"Expression: {expression}")
                print(f"Original indices: {idiom_indices}")
                print(f"Valid indices: {valid_indices}")
                continue
                
            for i, idx in enumerate(sorted(valid_indices)):
                bio_tags[idx] = 1 if i == 0 else 2
                label_counts[bio_tags[idx]] += 1
        
        label_counts[0] += bio_tags.count(0)
        inputs.append(sentence)
        labels.append(bio_tags)

    print("\nLabel distribution in dataset:")
    total = sum(label_counts.values())
    print(f"Total tokens: {total}")
    print(f"O (non-idiom): {label_counts[0]} ({label_counts[0]/total*100:.2f}%)")
    print(f"B-IDIOM: {label_counts[1]} ({label_counts[1]/total*100:.2f}%)")
    print(f"I-IDIOM: {label_counts[2]} ({label_counts[2]/total*100:.2f}%)")

    if not inputs:
        raise ValueError("No valid examples were processed!")

    return inputs, labels

def debug_data_loader(train_loader, tokenizer):
    print("\nDebugging first batch:")
    for batch in train_loader:
        input_ids = batch["input_ids"][0].tolist()  
        labels = batch["labels"][0].tolist()
        attention_mask = batch["attention_mask"][0].tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print("\nTokenized Sentence:")
        print(tokens)
        print("\nAttention Mask:")
        print(attention_mask)
        print("\nLabels:")
        print(labels)

        print("\nAligned tokens and labels:")
        label_counts = {0: 0, 1: 0, 2: 0}
        for token, label, mask in zip(tokens, labels, attention_mask):
            if mask:
                label_counts[label] += 1
            print(f"{token}: {label}")
        
        print("\nLabel distribution in batch:")
        total = sum(label_counts.values())
        print(f"O (non-idiom): {label_counts[0]} ({label_counts[0]/total*100:.2f}%)")
        print(f"B-IDIOM: {label_counts[1]} ({label_counts[1]/total*100:.2f}%)")
        print(f"I-IDIOM: {label_counts[2]} ({label_counts[2]/total*100:.2f}%)")
        break

def get_dataloaders(train_path="data/train.csv", val_path="data/eval.csv", batch_size=32, max_length=128, tokenizer=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")

    print(f"\nLoading data from:")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    print(f"\nDataset sizes:")
    print(f"Train: {len(df_train)} examples")
    print(f"Val: {len(df_val)} examples")

    print("\nProcessing training data...")
    train_inputs, train_labels = preprocess_data(df_train, tokenizer)
    print(f"Processed {len(train_inputs)} training examples")

    print("\nProcessing validation data...")
    val_inputs, val_labels = preprocess_data(df_val, tokenizer)
    print(f"Processed {len(val_inputs)} validation examples")

    print("\nCreating datasets...")
    train_dataset = IdiomDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = IdiomDataset(val_inputs, val_labels, tokenizer, max_length)

    print("\nCreating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nBatch size: {batch_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    return train_loader, val_loader