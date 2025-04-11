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

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move tensors to the configured device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)
            
            # Only consider tokens that aren't padding
            active_accuracy = attention_mask.view(-1) == 1
            active_preds = preds.view(-1)[active_accuracy]
            active_labels = labels.view(-1)[active_accuracy]
            
            all_preds.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)
    
    return val_loss / len(dataloader), accuracy, f1, report

def train_model(train_loader, val_loader, epochs=5, lr=2e-5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = BertForIdiomDetection()
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    best_f1 = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, accuracy, f1, report = evaluate(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "bert_model.pt")
            print("New best model saved!")
            print(report)
    
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
