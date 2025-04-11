import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from torchcrf import CRF


class BertForIdiomDetection(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=2):
        super(BertForIdiomDetection, self).__init__()
        
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Add a BiLSTM layer to capture context
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Classification layers
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(512, 256)  # 512 = 2*256 (bidirectional)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get token-level representations
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, 2*hidden_size]
        
        # Apply classification layers
        x = self.dropout(lstm_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Create mask for CRF
            crf_mask = attention_mask.bool()
            
            # CRF loss (negative log-likelihood)
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
        
        # CRF decoding for predictions
        if self.training or labels is None:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            # Convert list of lists to tensor with padding
            max_len = emissions.size(1)
            pred_tensor = torch.zeros_like(input_ids)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        else:
            # During evaluation, use argmax for simplicity in metrics calculation
            predictions = torch.argmax(emissions, dim=2)
            pred_tensor = predictions
        
        return {
            'loss': loss,
            'logits': emissions,
            'predictions': pred_tensor
        }

def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    val_loss = 0.0
    predictions = []
    ground_truth = []
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Skip empty batches
            if batch['input_ids'].size(0) == 0:
                continue
                
            total_batches += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Accumulate loss
            if outputs['loss'] is not None:
                val_loss += outputs['loss'].item()
            
            # Get predictions (now directly from model output)
            preds = outputs['predictions']
            
            # Process each sequence in batch
            for seq_preds, seq_mask, seq_labels, seq_ids in zip(preds, attention_mask, labels, input_ids):
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(seq_ids)
                
                # Map token predictions to word indices
                word_idx = -1
                current_word_preds = []
                word_level_preds = []
                
                for i, (token, mask, pred) in enumerate(zip(tokens, seq_mask, seq_preds)):
                    if mask == 0:  # Skip padding
                        continue
                        
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                        
                    if not token.startswith('##'):  # New word
                        # Process previous word if any
                        if current_word_preds:
                            if 1 in current_word_preds:
                                word_level_preds.append(word_idx)
                        
                        # Move to next word
                        word_idx += 1
                        current_word_preds = [pred.item()]
                    else:  # Continue current word
                        current_word_preds.append(pred.item())
                
                # Don't forget the last word
                if current_word_preds and 1 in current_word_preds:
                    word_level_preds.append(word_idx)
                
                # Extract ground truth word indices
                true_word_indices = []
                word_idx = -1
                
                for i, (token, mask, label) in enumerate(zip(tokens, seq_mask, seq_labels)):
                    if mask == 0:  # Skip padding
                        continue
                        
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                        
                    if not token.startswith('##'):  # New word
                        word_idx += 1
                        if label.item() == 1:
                            if word_idx not in true_word_indices:
                                true_word_indices.append(word_idx)
                
                # Store results
                predictions.append(word_level_preds)
                ground_truth.append(true_word_indices)
                
                # Debug print for a few examples
                if len(predictions) <= 5:
                    print("\nExample:")
                    print("Tokens:", tokens)
                    print("Token predictions:", seq_preds.tolist())
                    print("Mapped to word indices:", word_level_preds)
                    print("True indices:", true_word_indices)
    
    # Calculate average loss
    avg_val_loss = val_loss / max(1, total_batches)
    
    # Calculate F1 scores using competition method
    f1_scores = []
    for pred, gold in zip(predictions, ground_truth):
        # Handle special case for no idiom ([-1])
        if gold == [-1]:
            if pred == [-1] or not pred:  # Empty prediction is correct for no idiom
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
            continue
            
        # Normal case - set comparison
        pred_set = set(pred)
        gold_set = set(gold)
        
        intersection = len(pred_set & gold_set)
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
    
    mean_f1 = sum(f1_scores) / max(1, len(f1_scores))
    
    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    
    return {
        'loss': avg_val_loss,
        'f1': mean_f1,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

def train_model(train_loader, val_loader, tokenizer, epochs=10, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForIdiomDetection().to(device)
    
    # Differential learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' in n],
            'weight_decay': 0.01,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' in n],
            'weight_decay': 0.0,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' not in n],
            'weight_decay': 0.01,
            'lr': lr * 10  # Higher learning rate for custom layers
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' not in n],
            'weight_decay': 0.0,
            'lr': lr * 10  # Higher learning rate for custom layers
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[lr, lr, lr*10, lr*10],
        total_steps=total_steps
    )
    
    best_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        metrics = evaluate(model, val_loader, tokenizer, device)
        
        print(f"Epoch {epoch+1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "best_idiom_model.pt")
            print("New best model saved!")
    
    return model

def predict_idioms(model, tokenizer, sentence, device):
    model.eval()
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Get predictions from CRF
    predictions = outputs['predictions'][0]
    
    # Map to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Convert to word-level predictions
    word_preds = []
    current_word = ""
    current_pred = 0
    word_idx = -1
    words = []
    
    for token, pred, mask in zip(tokens, predictions, attention_mask[0]):
        if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if not token.startswith('##'):
            # Save previous word
            if current_word:
                words.append(current_word)
                if current_pred == 1:
                    word_preds.append(word_idx)
                
            # Start new word
            word_idx += 1
            current_word = token
            current_pred = pred.item()
        else:
            # Continue current word
            current_word += token[2:]  # Remove ## prefix
            current_pred = max(current_pred, pred.item())
    
    # Don't forget last word
    if current_word:
        words.append(current_word)
        if current_pred == 1:
            word_preds.append(word_idx)
    
    # Format results
    results = [(word, 1 if i in word_preds else 0) for i, word in enumerate(words)]
    return results

def debug_predictions(model, tokenizer, test_sentences, device):
    """
    Debug function to show the complete pipeline of tokenization, prediction, and remapping
    """
    model.eval()
    
    for sentence in test_sentences:
        print("\n" + "="*80)
        print(f"Original sentence: {sentence}")
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2)
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Map predictions back to words
        word_idx = -1  # Start at -1 to account for [CLS]
        current_word_preds = []
        word_level_preds = []
        words = []
        current_word = []
        
        print("\nDetailed token analysis:")
        print(f"{'Token':<15} {'Is Subword':<12} {'Prediction':<10} {'Word Index':<10}")
        print("-" * 50)
        
        for token, pred in zip(tokens, preds[0]):
            if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                print(f"{token:<15} {'N/A':<12} {pred.item():<10} {'N/A':<10}")
                continue
                
            is_subword = token.startswith('##')
            
            if not is_subword:  # New word
                # Save prediction for previous word
                if current_word_preds:
                    if 1 in current_word_preds:
                        word_level_preds.append(word_idx)
                    words.append(''.join(current_word))
                word_idx += 1
                current_word_preds = [pred.item()]
                current_word = [token]
            else:  # Subword
                current_word_preds.append(pred.item())
                current_word.append(token[2:])  # Remove ## prefix
                
            print(f"{token:<15} {str(is_subword):<12} {pred.item():<10} {word_idx:<10}")
        
        # Handle last word
        if current_word_preds and 1 in current_word_preds:
            word_level_preds.append(word_idx)
        if current_word:
            words.append(''.join(current_word))
            
        print("\nFinal Analysis:")
        print("Reconstructed words:", words)
        print("Word-level predictions:", word_level_preds)
        print("Predicted idiom words:", [words[i] for i in word_level_preds])
