import pandas as pd
import torch
from transformers import BertTokenizer
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

        # Tokenize and pad inputs
        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Add padding to labels to account for [CLS] and [SEP] tokens
        label = [0] + label + [0]  
        
        # Pad labels to match max length
        label = label + [0] * (self.max_length - len(label))
        label = torch.tensor(label[:self.max_length], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }

def preprocess_data(df, tokenizer):
    inputs, labels = [], []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        idiom_indices = eval(row["indices"])  # Convert "[0, 1]" -> [0, 1]
        tokenized_words = eval(row["tokenized_sentence"])  # Get original words
        idiom_label_map = {idx: 1 for idx in idiom_indices}  # Mark idiom positions

        # Tokenize the sentence with WordPiece
        wordpiece_tokens = []
        label_list = []

        for word_idx, word in enumerate(tokenized_words):
            subwords = tokenizer.tokenize(word)  # Get subword tokens
            wordpiece_tokens.extend(subwords)

            # Assign the same label to all subwords of the idiomatic word
            label = idiom_label_map.get(word_idx, 0)  # Default is 0 (non-idiom)
            label_list.extend([label] * len(subwords))  # Extend to all subwords
        
        inputs.append(wordpiece_tokens)
        labels.append(label_list)

    return inputs, labels

def debug_data_loader(train_loader, tokenizer):
    for batch in train_loader:
        input_ids = batch["input_ids"][0].tolist()  
        labels = batch["labels"][0].tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print("\nTokenized Sentence:")
        print(tokens)

        print("\nLabels:")
        print(labels)

        print("\nLabeled Tokens (Idioms should be 1):")
        for token, label in zip(tokens, labels):
            print(f"{token}: {label}")
        break  

def get_dataloaders(train_path="data/train.csv", val_path="data/eval.csv", batch_size=8, max_length=128):

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

  
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


    train_inputs, train_labels = preprocess_data(df_train, tokenizer)
    val_inputs, val_labels = preprocess_data(df_val, tokenizer)


    train_dataset = IdiomDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = IdiomDataset(val_inputs, val_labels, tokenizer, max_length)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer