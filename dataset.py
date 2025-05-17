import pandas as pd
import torch
from transformers import AutoTokenizer
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
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )


        if len(label) > self.max_length - 2:  # Account for [CLS] and [SEP]
            label = label[:self.max_length - 2]
            
        # Add [CLS] and [SEP] token labels (always 0 = 'O' tag)
        padded_label = [0] + label + [0]
        
        # Pad the rest with 0s
        padded_label = padded_label + [0] * (self.max_length - len(padded_label))
        
        padded_label = torch.tensor(padded_label[:self.max_length], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": padded_label
        }

def preprocess_data(df, tokenizer):
    inputs = []
    labels = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        idiom_indices = eval(row["indices"])  # Convert "[0, 1]" -> [0, 1]
        
        # Special case for no idiom
        if idiom_indices == [-1]:
            # Process the full sentence and assign all "O" tags
            inputs.append(sentence)
            # Create dummy labels that will be adjusted in __getitem__
            tokens = tokenizer.tokenize(sentence)
            label_list = [0] * len(tokens)  # 0 = O tag
            labels.append(label_list)
            continue
        
        tokenized_words = eval(row["tokenized_sentence"])
        
        # Create BIO tags: 0=O, 1=B-IDIOM, 2=I-IDIOM
        bio_tags = [0] * len(tokenized_words)  # Initialize all as O
        
        # Process idiom indices using BIO scheme
        for i, idx in enumerate(sorted(idiom_indices)):
            if i == 0:  # First token of idiom
                bio_tags[idx] = 1  # B-IDIOM
            else:
                bio_tags[idx] = 2  # I-IDIOM
                
        # Special case: non-consecutive indices 
        # Check if there are gaps and handle them
        for i in range(len(idiom_indices) - 1):
            if idiom_indices[i+1] - idiom_indices[i] > 1:
                # If gap, the next token should be B-IDIOM, not I-IDIOM
                bio_tags[idiom_indices[i+1]] = 1
        
        # Store the original sentence
        inputs.append(sentence)
        
        # Pre-tokenize to get accurate labels
        # For each word, find how it gets tokenized and adjust labels
        tokenizer_outputs = tokenizer(sentence, add_special_tokens=False)
        input_ids = tokenizer_outputs["input_ids"]
        
        # Get all tokens and determine alignment
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Re-tokenize each word in tokenized_words to build alignment
        word_to_tokens_map = []
        token_idx = 0
        
        # Attempt to align the pre-tokenized words with the tokenizer's output
        label_list = []
        for word_idx, word in enumerate(tokenized_words):
            # Get the label for this word
            word_label = bio_tags[word_idx]
            
            # Find how many tokens this word was split into
            word_tokens = tokenizer.tokenize(word)
            num_tokens = len(word_tokens)
            
            # First token gets the original label
            label_list.append(word_label)
            
            # All subsequent tokens for this word get:
            # - I-IDIOM (2) if original was B-IDIOM (1)
            # - same label otherwise
            if num_tokens > 1:
                if word_label == 1:  # B-IDIOM
                    label_list.extend([2] * (num_tokens - 1))  # Rest are I-IDIOM
                else:
                    label_list.extend([word_label] * (num_tokens - 1))
        
        # Store the aligned labels
        labels.append(label_list)

    return inputs, labels

def get_dataloaders(train_path="data/train.csv", val_path="data/eval.csv", batch_size=8, max_length=128, model_name="bert-base-multilingual-cased"):

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

  
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    train_inputs, train_labels = preprocess_data(df_train, tokenizer)
    val_inputs, val_labels = preprocess_data(df_val, tokenizer)


    train_dataset = IdiomDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = IdiomDataset(val_inputs, val_labels, tokenizer, max_length)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer