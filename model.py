import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from torchcrf import CRF


class IdiomDetectionModel(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=3):  # 3 labels: O, B-IDIOM, I-IDIOM
        super(IdiomDetectionModel, self).__init__()
        
        # Pre-trained model using AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Add a BiLSTM layer to capture context
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Add multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # 2*256 from BiLSTM
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification layers
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(512, 256)  # 512 = 2*256 (bidirectional)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Focal loss parameters
        self.alpha = 2.0  # Focusing parameter
        self.gamma = 0.25  # Class balancing parameter
        
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
        
        # Apply multi-head attention
        attn_output, _ = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=~attention_mask.bool()  # Convert mask: 1->valid, 0->padding
        )
        
        # Add residual connection from LSTM output to attention output
        combined = attn_output + lstm_output
        
        # Apply classification layers
        x = self.dropout(combined)
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Create mask for CRF
            crf_mask = attention_mask.bool()
            
            # Standard CRF loss (negative log-likelihood)
            crf_loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
            
            # Add focal loss component
            # Compute softmax probabilities
            probs = torch.softmax(emissions, dim=-1)
            # Get probability of correct class
            pt = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            # Apply focusing parameter
            focal_weight = (1 - pt) ** self.alpha
            # Compute cross entropy loss
            ce_loss = F.cross_entropy(
                emissions.view(-1, emissions.size(-1)), 
                labels.view(-1), 
                reduction='none'
            )
            # Apply mask to ignore padding
            ce_loss = ce_loss.view(emissions.size(0), emissions.size(1))
            masked_ce_loss = ce_loss * attention_mask.float()
            # Apply focal weighting
            focal_weight = focal_weight * attention_mask.float()
            focal_loss = (focal_weight * masked_ce_loss).sum() / attention_mask.sum()
            
            # Combined loss
            loss = crf_loss + self.gamma * focal_loss
        
        # CRF decoding for predictions
        if self.training or labels is None:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            # Convert list of lists to tensor with padding
            max_len = emissions.size(1)
            pred_tensor = torch.zeros_like(input_ids)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        else:
            # During evaluation, use CRF decoding
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            # Convert list of lists to tensor with padding
            max_len = emissions.size(1)
            pred_tensor = torch.zeros_like(input_ids)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        
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
            
            if outputs['loss'] is not None:
                val_loss += outputs['loss'].item()
            
            preds = outputs['predictions']
            
            # Process each sequence in batch
            for seq_preds, seq_mask, seq_labels, seq_ids in zip(preds, attention_mask, labels, input_ids):
                # Only process sequences with valid inputs
                if seq_mask.sum() == 0:
                    continue
                    
                tokens = tokenizer.convert_ids_to_tokens(seq_ids)
                
                # Extract idiom indices based on BIO tags
                # For ground truth
                word_idx = -1
                true_idiom_indices = []
                current_idiom_indices = []
                previous_tag = 0  # O tag
                
                for i, (token, mask, label) in enumerate(zip(tokens, seq_mask, seq_labels)):
                    if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    
                    # Handle different tokenizer special tokens
                    is_subword = False
                    if token.startswith('##'):  # BERT style
                        is_subword = True
                    elif token.startswith('▁'):  # XLM-R style
                        is_subword = False
                    elif i > 0 and not token.startswith('Ġ') and not tokens[i-1] in ['[CLS]', '[SEP]', '[PAD]']:
                        # RoBERTa, DeBERTa style subwords don't have a prefix if they continue a word
                        is_subword = True
                        
                    if not is_subword:  # New word
                        word_idx += 1
                        
                        # Handle end of previous idiom
                        if previous_tag in [1, 2] and label.item() not in [1, 2]:
                            # End of idiom
                            if current_idiom_indices:
                                true_idiom_indices.extend(current_idiom_indices)
                                current_idiom_indices = []
                        
                        # Handle new idiom
                        if label.item() == 1:  # B-IDIOM
                            current_idiom_indices = [word_idx]
                        elif label.item() == 2:  # I-IDIOM
                            if previous_tag in [1, 2]:  # Continue idiom
                                current_idiom_indices.append(word_idx)
                        
                        previous_tag = label.item()
                
                # Don't forget last idiom
                if current_idiom_indices:
                    true_idiom_indices.extend(current_idiom_indices)
                
                # For predictions
                word_idx = -1
                pred_idiom_indices = []
                current_idiom_indices = []
                previous_tag = 0  # O tag
                
                for i, (token, mask, pred) in enumerate(zip(tokens, seq_mask, seq_preds)):
                    if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                    
                    # Handle different tokenizer special tokens
                    is_subword = False
                    if token.startswith('##'):  # BERT style
                        is_subword = True
                    elif token.startswith('▁'):  # XLM-R style
                        is_subword = False
                    elif i > 0 and not token.startswith('Ġ') and not tokens[i-1] in ['[CLS]', '[SEP]', '[PAD]']:
                        # RoBERTa, DeBERTa style subwords don't have a prefix if they continue a word
                        is_subword = True
                        
                    if not is_subword:  # New word
                        word_idx += 1
                        
                        # Handle end of previous idiom
                        if previous_tag in [1, 2] and pred.item() not in [1, 2]:
                            # End of idiom
                            if current_idiom_indices:
                                pred_idiom_indices.extend(current_idiom_indices)
                                current_idiom_indices = []
                        
                        # Handle new idiom
                        if pred.item() == 1:  # B-IDIOM
                            current_idiom_indices = [word_idx]
                        elif pred.item() == 2:  # I-IDIOM
                            if previous_tag in [1, 2]:  # Continue idiom
                                current_idiom_indices.append(word_idx)
                        
                        previous_tag = pred.item()
                
                # Don't forget last idiom
                if current_idiom_indices:
                    pred_idiom_indices.extend(current_idiom_indices)
                
                # Store results
                predictions.append(pred_idiom_indices)
                ground_truth.append(true_idiom_indices)
                
                # Debug print for a few examples
                if len(predictions) <= 5:
                    print("\nExample:")
                    print("Tokens:", tokens)
                    print("True BIO tags:", seq_labels.tolist())
                    print("Pred BIO tags:", seq_preds.tolist())
                    print("True idiom indices:", true_idiom_indices)
                    print("Pred idiom indices:", pred_idiom_indices)
    
    # Calculate average loss
    avg_val_loss = val_loss / max(1, total_batches)
    
    # Calculate F1 scores using competition method
    f1_scores = []
    for pred, gold in zip(predictions, ground_truth):
        # Handle special case for no idiom
        if not gold:  # Empty gold = no idiom
            if not pred:  # Empty pred = correctly predicted no idiom
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
    model = IdiomDetectionModel().to(device)
    
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
    preds = outputs['predictions'][0]
    
    # Map to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    masks = attention_mask[0]
    
    # Extract idioms based on BIO tags
    words = []
    bio_tags = []
    word_idx = -1
    idiom_indices = []
    current_idiom = []
    current_word = ""
    previous_tag = 0  # O tag
    
    # Get tokenizer type for handling subwords
    if hasattr(tokenizer, 'name_or_path'):
        tokenizer_name = tokenizer.name_or_path.lower()
    else:
        tokenizer_name = type(tokenizer).__name__.lower()
    
    is_bert_type = 'bert' in tokenizer_name
    is_roberta_type = 'roberta' in tokenizer_name or 'deberta' in tokenizer_name
    is_xlm_type = 'xlm' in tokenizer_name
    
    for i, (token, mask, pred) in enumerate(zip(tokens, masks, preds)):
        if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue
            
        # Check if this is a subword token based on tokenizer type
        is_subword = False
        if is_bert_type and token.startswith('##'):
            is_subword = True
        elif is_xlm_type and token.startswith('▁') and i > 0:
            is_subword = False  # XLM-R uses ▁ for start of new word
        elif is_roberta_type and i > 0 and not token.startswith('Ġ') and tokens[i-1] not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            is_subword = True
            
        if not is_subword:  # New word
            # Save previous word
            if current_word:
                words.append(current_word)
                bio_tags.append(previous_tag)
                word_idx += 1
                
                # Handle idiom tracking
                if previous_tag in [1, 2] and pred.item() not in [1, 2]:  # End of idiom
                    if current_idiom:
                        idiom_indices.extend(current_idiom)
                        current_idiom = []
            
            # Start new word
            if is_xlm_type and token.startswith('▁'):
                current_word = token[1:]  # Remove ▁ prefix for XLM-R
            elif is_roberta_type and token.startswith('Ġ'):
                current_word = token[1:]  # Remove Ġ prefix for RoBERTa
            else:
                current_word = token
                
            previous_tag = pred.item()
            
            # Track idioms
            if pred.item() == 1:  # B-IDIOM
                current_idiom = [word_idx + 1]  # +1 because we haven't incremented yet
            elif pred.item() == 2:  # I-IDIOM
                if previous_tag in [1, 2]:  # Continue idiom
                    current_idiom.append(word_idx + 1)
        else:
            # Continue current word
            if is_bert_type and token.startswith('##'):
                current_word += token[2:]  # Remove ## prefix for BERT
            else:
                current_word += token  # No prefix for other tokenizers' subwords
    
    # Don't forget last word
    if current_word:
        words.append(current_word)
        bio_tags.append(previous_tag)
        
        # Handle last idiom
        if previous_tag in [1, 2] and current_idiom:
            idiom_indices.extend(current_idiom)
    
    # Format results with BIO tags
    results = []
    for i, (word, tag) in enumerate(zip(words, bio_tags)):
        if tag == 0:
            results.append((word, "O"))
        elif tag == 1:
            results.append((word, "B-IDIOM"))
        elif tag == 2:
            results.append((word, "I-IDIOM"))
    
    return results, idiom_indices

def debug_predictions(model, tokenizer, test_sentences, device):
    """
    Debug function to show the complete pipeline of tokenization, prediction, and remapping
    """
    model.eval()
    
    # Get tokenizer type for handling subwords
    if hasattr(tokenizer, 'name_or_path'):
        tokenizer_name = tokenizer.name_or_path.lower()
    else:
        tokenizer_name = type(tokenizer).__name__.lower()
    
    is_bert_type = 'bert' in tokenizer_name
    is_roberta_type = 'roberta' in tokenizer_name or 'deberta' in tokenizer_name
    is_xlm_type = 'xlm' in tokenizer_name
    
    # Get tokenizer's special tokens
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    special_tokens = [cls_token, sep_token, pad_token]
    
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
        word_idx = -1  # Start at -1 to account for special tokens
        current_word_preds = []
        word_level_preds = []
        words = []
        current_word = []
        
        print("\nDetailed token analysis:")
        print(f"{'Token':<15} {'Is Subword':<12} {'Prediction':<10} {'Word Index':<10}")
        print("-" * 50)
        
        for i, (token, pred) in enumerate(zip(tokens, preds[0])):
            if token in special_tokens:
                print(f"{token:<15} {'N/A':<12} {pred.item():<10} {'N/A':<10}")
                continue
            
            # Check if this is a subword token based on tokenizer type
            is_subword = False
            if is_bert_type and token.startswith('##'):
                is_subword = True
            elif is_xlm_type and token.startswith('▁') and i > 0:
                is_subword = False  # XLM-R uses ▁ for start of new word
            elif is_roberta_type and i > 0 and not token.startswith('Ġ') and tokens[i-1] not in special_tokens:
                is_subword = True
                
            if not is_subword:  # New word
                # Save prediction for previous word
                if current_word_preds:
                    if 1 in current_word_preds:
                        word_level_preds.append(word_idx)
                    if current_word:
                        words.append(''.join(current_word))
                word_idx += 1
                current_word_preds = [pred.item()]
                
                # Process the token based on tokenizer type
                if is_xlm_type and token.startswith('▁'):
                    current_word = [token[1:]]  # Remove ▁ prefix for XLM-R
                elif is_roberta_type and token.startswith('Ġ'):
                    current_word = [token[1:]]  # Remove Ġ prefix for RoBERTa
                else:
                    current_word = [token]
            else:  # Subword
                current_word_preds.append(pred.item())
                if is_bert_type and token.startswith('##'):
                    current_word.append(token[2:])  # Remove ## prefix for BERT
                else:
                    current_word.append(token)  # No prefix removal for other tokenizers
                
            print(f"{token:<15} {str(is_subword):<12} {pred.item():<10} {word_idx:<10}")
        
        # Handle last word
        if current_word_preds and 1 in current_word_preds:
            word_level_preds.append(word_idx)
        if current_word:
            words.append(''.join(current_word))
            
        print("\nFinal Analysis:")
        print("Reconstructed words:", words)
        print("Word-level predictions:", word_level_preds)
        print("Predicted idiom words:", [words[i] for i in word_level_preds if i < len(words)])
