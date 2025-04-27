import torch
import torch.nn as nn
import math
from torchcrf import CRF
from tqdm import tqdm
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CustomTransformerForIdiomDetection(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, num_labels=3):
        super().__init__()
        
        # Core components
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(0.4)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Feature extraction and processing
        self.feature_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.3)
            ) for k in [3, 5, 7]
        ])
        
        # Transformer and LSTM layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.2,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=4
        )
        
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_labels)
        )
        
        self.crf = CRF(num_labels, batch_first=True)
        self.mixup_alpha = 0.2
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def forward(self, input_ids, attention_mask, labels=None):
        # Initial embeddings
        x = self.word_embedding(input_ids)
        x = self.embedding_norm(x)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        
        # Apply mixup during training
        if self.training and labels is not None:
            x, labels_a, labels_b, lam = self.mixup_data(x, labels, self.mixup_alpha)
        
        # Feature extraction
        features = []
        x_conv = x.transpose(1, 2)
        for conv in self.feature_processor:
            features.append(conv(x_conv).transpose(1, 2))
        x = torch.stack(features).mean(0)
        
        # Transformer and LSTM processing
        mask = attention_mask.bool()
        x = self.transformer(x, src_key_padding_mask=~mask)
        lstm_out, _ = self.bilstm(x)
        
        # Classification
        x = torch.cat([x, lstm_out], dim=-1)
        emissions = self.classifier(x)
        
        outputs = {'logits': emissions}
        
        if labels is not None:
            if self.training:
                # Training loss
                crf_loss_a = -self.crf(emissions, labels_a, mask=mask, reduction='mean')
                crf_loss_b = -self.crf(emissions, labels_b, mask=mask, reduction='mean')
                loss = lam * crf_loss_a + (1 - lam) * crf_loss_b
            else:
                # Validation loss
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            outputs['loss'] = loss
        
        # Predictions
        if not self.training:
            predictions = self.crf.decode(emissions, mask=mask)
            max_len = max(len(seq) for seq in predictions)
            padded_predictions = torch.zeros((input_ids.size(0), max_len), dtype=torch.long, device=emissions.device)
            for i, seq in enumerate(predictions):
                padded_predictions[i, :len(seq)] = torch.tensor(seq, device=emissions.device)
            outputs['predictions'] = padded_predictions
        
        return outputs

def train_custom_model(train_loader, val_loader, tokenizer, epochs=50, lr=3e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    model = CustomTransformerForIdiomDetection(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=len(train_loader) * epochs,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    best_f1 = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs['loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
        
        print(f"Average training loss: {train_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_metrics = {'loss': 0, 'true': [], 'pred': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_metrics['loss'] += outputs['loss'].item()
                
                mask = batch['attention_mask'].bool()
                for pred, label, length in zip(outputs['predictions'], batch['labels'], mask.sum(1)):
                    val_metrics['pred'].extend(pred[:length].cpu().tolist())
                    val_metrics['true'].extend(label[:length].cpu().tolist())
        
        # Calculate metrics
        val_metrics['loss'] /= len(val_loader)
        y_true = torch.tensor(val_metrics['true'])
        y_pred = torch.tensor(val_metrics['pred'])
        mask_idiom = (y_true != 0) | (y_pred != 0)
        
        if mask_idiom.sum() > 0:
            precision = (y_true[mask_idiom] == y_pred[mask_idiom]).float().mean().item()
            recall = ((y_true != 0) & (y_true == y_pred)).float().sum() / (y_true != 0).float().sum().item()
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0.0
        
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model 