import torch
import torch.nn as nn
import math
from torchcrf import CRF
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Optional, Tuple
from gensim.models import KeyedVectors
import logging
import gdown
import zipfile

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output(context)
        
        return output, attn_weights

class PretrainedEmbeddings:
    GOOGLE_WORD2VEC_URL = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    
    @staticmethod
    def download_google_word2vec(save_dir: str = "./embeddings") -> str:
        """
        Download Google's pre-trained Word2Vec model if not already present.
        
        Args:
            save_dir: Directory to save the model
        Returns:
            Path to the Word2Vec binary file
        """
        os.makedirs(save_dir, exist_ok=True)
        zip_path = os.path.join(save_dir, "GoogleNews-vectors-negative300.bin.gz")
        model_path = os.path.join(save_dir, "GoogleNews-vectors-negative300.bin")
        
        if not os.path.exists(model_path):
            if not os.path.exists(zip_path):
                logging.info("Downloading Google's pre-trained Word2Vec model...")
                gdown.download(PretrainedEmbeddings.GOOGLE_WORD2VEC_URL, zip_path, quiet=False)
            
            logging.info("Extracting Word2Vec model...")
            import gzip
            with gzip.open(zip_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            logging.info(f"Model extracted to {model_path}")
        
        return model_path

    @staticmethod
    def load_word2vec_embeddings(path: str, limit: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Load Word2Vec embeddings from a .bin or .txt file.
        
        Args:
            path: Path to Word2Vec embeddings file (binary or text format)
            limit: Optional limit on number of words to load (useful for testing)
        Returns:
            Tuple of (embeddings dictionary, embedding dimension)
        """
        logging.info(f"Loading Word2Vec embeddings from {path}")
        # Load the Word2Vec model
        word2vec = KeyedVectors.load_word2vec_format(
            path, 
            binary=path.endswith('.bin'),
            limit=limit
        )
        
        # Convert to dictionary of PyTorch tensors
        embeddings_dict = {}
        for word in word2vec.index_to_key:
            embeddings_dict[word] = torch.FloatTensor(word2vec[word])
        
        embedding_dim = len(next(iter(embeddings_dict.values())))
        logging.info(f"Loaded {len(embeddings_dict)} word vectors of dimension {embedding_dim}")
        return embeddings_dict, embedding_dim

    @staticmethod
    def create_embedding_matrix(word_to_idx: Dict[str, int], embeddings_dict: Dict[str, torch.Tensor], 
                              embedding_dim: int) -> torch.Tensor:
        """
        Create an embedding matrix for the vocabulary using pre-trained embeddings.
        
        Args:
            word_to_idx: Dictionary mapping words to indices
            embeddings_dict: Dictionary of pre-trained word embeddings
            embedding_dim: Dimension of the embeddings
        Returns:
            Tensor containing the embedding matrix
        """
        vocab_size = len(word_to_idx)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
        embedding_matrix.normal_(0, 0.1)  # Initialize unknown tokens
        
        num_pretrained = 0
        for word, idx in word_to_idx.items():
            if word in embeddings_dict:
                embedding_matrix[idx] = embeddings_dict[word]
                num_pretrained += 1
        
        coverage = num_pretrained / vocab_size * 100
        logging.info(f"Initialized embeddings matrix with {coverage:.2f}% coverage ({num_pretrained}/{vocab_size} words)")
        return embedding_matrix

    @classmethod
    def from_google_word2vec(cls, vocab_size: int, word_to_idx: Dict[str, int], 
                            save_dir: str = "./embeddings", limit: Optional[int] = None,
                            **kwargs):
        """
        Initialize model with Google's pre-trained Word2Vec embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            word_to_idx: Dictionary mapping words to indices
            save_dir: Directory to save the downloaded model
            limit: Optional limit on number of words to load (useful for testing)
            **kwargs: Additional arguments for model initialization
        """
        # Download and load Google's Word2Vec embeddings
        model_path = PretrainedEmbeddings.download_google_word2vec(save_dir)
        embeddings_dict, embedding_dim = PretrainedEmbeddings.load_word2vec_embeddings(
            model_path, limit=limit
        )
        
        weights = PretrainedEmbeddings.create_embedding_matrix(
            word_to_idx, embeddings_dict, embedding_dim
        )
        
        return cls(
            vocab_size=vocab_size,
            d_model=embedding_dim,  # Google's Word2Vec uses 300 dimensions
            pretrained_embeddings=weights,
            **kwargs
        )

class CustomTransformerForIdiomDetection(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, num_labels: int = 3, 
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        
        # Core components with pre-trained embeddings support
        if pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,  # Allow fine-tuning
                padding_idx=0
            )
            logging.info(f"Initialized embedding layer with pretrained weights of shape {pretrained_embeddings.shape}")
        else:
            self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            logging.info(f"Initialized random embedding layer of shape ({vocab_size}, {d_model})")
            
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
        
        # Transformer encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Cross attention between transformer and LSTM
        self.cross_attention = MultiHeadAttention(d_model, nhead)
        
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )
        
        # Residual and layer norm components
        self.residual_norm1 = nn.LayerNorm(d_model)
        self.residual_norm2 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model * 2)
        
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

    @classmethod
    def from_pretrained(cls, vocab_size: int, word_to_idx: Dict[str, int], 
                       embeddings_path: str, d_model: Optional[int] = None, **kwargs):
        """
        Initialize model with pre-trained Word2Vec embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            word_to_idx: Dictionary mapping words to indices
            embeddings_path: Path to Word2Vec embeddings file (.bin or .txt format)
            d_model: Model dimension (if None, will use the dimension of loaded embeddings)
            **kwargs: Additional arguments for model initialization
        """
        # Load pre-trained embeddings
        embeddings_dict = PretrainedEmbeddings.load_word2vec_embeddings(embeddings_path)
        
        # Get embedding dimension from loaded embeddings if not specified
        if d_model is None:
            d_model = len(next(iter(embeddings_dict.values())))
            logging.info(f"Setting d_model to embedding dimension: {d_model}")
        
        weights = PretrainedEmbeddings.create_embedding_matrix(
            word_to_idx, embeddings_dict, d_model
        )
        
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            pretrained_embeddings=weights,
            **kwargs
        )

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
        
        # Store original input for residual connections
        input_embedding = x
        
        # Apply mixup during training
        if self.training and labels is not None:
            x, labels_a, labels_b, lam = self.mixup_data(x, labels, self.mixup_alpha)
        
        # Feature extraction
        features = []
        x_conv = x.transpose(1, 2)
        for conv in self.feature_processor:
            features.append(conv(x_conv).transpose(1, 2))
        x = torch.stack(features).mean(0)
        
        # Add residual connection from input embeddings
        x = x + input_embedding
        x = self.residual_norm1(x)
        
        # Transformer processing
        mask = attention_mask.bool()
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Decoder processing with residual connection
        decoder_output = self.transformer_decoder(
            x,
            encoder_output,
            tgt_key_padding_mask=~mask,
            memory_key_padding_mask=~mask
        )
        decoder_output = decoder_output + x
        decoder_output = self.residual_norm2(decoder_output)
        
        # LSTM processing
        lstm_out, _ = self.bilstm(decoder_output)
        
        # Cross attention between transformer decoder and LSTM outputs
        attended_output, _ = self.cross_attention(
            decoder_output,  # query
            lstm_out,        # key
            lstm_out,        # value
            mask=mask
        )
        
        # Concatenate and normalize
        x = torch.cat([attended_output, lstm_out], dim=-1)
        x = self.final_norm(x)
        
        # Classification
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