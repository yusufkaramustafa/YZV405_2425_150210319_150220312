import torch
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

class BertForIdiomDetection(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=2):
        super(BertForIdiomDetection, self).__init__()
        
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Classification head for token-level classification
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get token-level representations
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classify
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 2)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            
        return {
            'loss': loss,
            'logits': logits
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move tensors to the configured device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    val_loss = 0
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2)
            
            # Process each sequence in batch
            for seq_preds, seq_ids, seq_mask, seq_labels in zip(preds, input_ids, attention_mask, labels):
                tokens = tokenizer.convert_ids_to_tokens(seq_ids)
                
                # Map subword predictions back to word-level predictions
                word_idx = -1  # Start at -1 to account for [CLS]
                current_word_preds = []
                word_level_preds = []
                
                for token, pred in zip(tokens, seq_preds):
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                        
                    if not token.startswith('##'):  # New word
                        # Save prediction for previous word
                        if current_word_preds:
                            # If any subword was predicted as idiom, whole word is idiom
                            if 1 in current_word_preds:
                                word_level_preds.append(word_idx)
                        word_idx += 1
                        current_word_preds = [pred.item()]
                    else:  # Subword
                        current_word_preds.append(pred.item())
                
                # The last word
                if current_word_preds and 1 in current_word_preds:
                    word_level_preds.append(word_idx)
                
                # Get ground truth indices
                true_indices = []
                word_idx = -1
                for token, label in zip(tokens, seq_labels):
                    if token == '[CLS]' or token == '[SEP]':
                        continue
                    if not token.startswith('##'):
                        word_idx += 1
                        if label.item() == 1:
                            true_indices.append(word_idx)
                
                predictions.append(word_level_preds)
                ground_truth.append(true_indices)
                
                # Debug print
                print("\nExample:")
                print("Tokens:", tokens)
                print("Token predictions:", seq_preds.tolist())
                print("Mapped to word indices:", word_level_preds)
                print("True indices:", true_indices)
    
    # Calculate F1 scores
    f1_scores = []
    for pred, gold in zip(predictions, ground_truth):
        pred_set = set(pred)
        gold_set = set(gold)
        intersection = len(pred_set & gold_set)
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    print(f"\nMean F1 Score: {mean_f1:.4f}")
    
    return {
        'f1': mean_f1,
        'loss': val_loss / len(dataloader),
        'predictions': predictions,
        'ground_truth': ground_truth
    }

def train_model(train_loader, val_loader, tokenizer, epochs=5, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForIdiomDetection().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_loader, tokenizer, device)
        
        # Save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "best_idiom_model.pt")
            print("New best model saved!")
            print(f"F1 Score: {metrics['f1']:.4f}")
    
    return model

def predict_idioms(model, tokenizer, sentence, device):
    """
    Predict idioms in a raw sentence
    
    Args:
        model: Trained model
        tokenizer: BERT tokenizer
        sentence: Raw text string
        device: torch device
        
    Returns:
        Tokens and their predicted labels
    """
    model.eval()
    
    # Tokenize input
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
        
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=2)
    
    # Extract tokens and their predictions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_predictions = predictions[0].cpu().numpy()
    
    results = []
    for token, pred, mask in zip(tokens, token_predictions, attention_mask[0]):
        if mask == 1:  # Skip padding
            results.append((token, int(pred)))
    
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
