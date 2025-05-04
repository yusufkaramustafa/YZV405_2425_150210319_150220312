import torch
import torch.nn as nn
import math
from torchcrf import CRF
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Optional, Tuple, List, Union
from gensim.models import KeyedVectors
import logging
import gdown
import zipfile
from transformers import BertModel, BertTokenizer, BertConfig
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class BertEmbeddings:
    """
    Class to handle BERT embeddings for the custom transformer model
    """
    @staticmethod
    def get_bert_model(model_name: str = "bert-base-multilingual-cased", num_hidden_layers: int = 4):
        """
        Load a pre-trained BERT model with a specific number of layers
        
        Args:
            model_name: Name of the BERT model to load
            num_hidden_layers: Number of transformer layers to use from BERT
        
        Returns:
            BERT model instance with truncated layers
        """
        logger.info(f"Loading BERT model: {model_name} with {num_hidden_layers} layers")
        
        # Load BERT configuration
        config = BertConfig.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # Modify config to use fewer layers if specified
        if num_hidden_layers < config.num_hidden_layers:
            config.num_hidden_layers = num_hidden_layers
            logger.info(f"Using truncated BERT with {num_hidden_layers} layers")
        
        # Load model with modified config
        model = BertModel.from_pretrained(model_name, config=config)
        return model
    
    @classmethod
    def create_bert_embedder(cls, model_name: str = "bert-base-multilingual-cased", num_layers: int = 4, freeze: bool = True):
        """
        Create a BERT embedder with specified parameters
        
        Args:
            model_name: Name of the BERT model to load
            num_layers: Number of transformer layers to use from BERT
            freeze: Whether to freeze the BERT parameters
            
        Returns:
            BERT embedder instance
        """
        bert_model = cls.get_bert_model(model_name, num_layers)
        
        # Freeze BERT parameters if specified
        if freeze:
            logger.info("Freezing BERT parameters")
            for param in bert_model.parameters():
                param.requires_grad = False
        
        return bert_model

class CustomTransformerForIdiomDetection(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 8, 
                 num_layers: int = 6, num_labels: int = 3, dropout: float = 0.2,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 use_bert: bool = True, bert_model_name: str = "bert-base-multilingual-cased",
                 bert_layers: int = 4, freeze_bert: bool = True,
                 enable_mlm: bool = False, mlm_probability: float = 0.15):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_labels = num_labels
        self.enable_mlm = enable_mlm
        self.mlm_probability = mlm_probability
        self.use_bert = use_bert
        
        # BERT or Word2Vec embeddings
        if use_bert:
            logger.info(f"Using BERT embeddings from {bert_model_name} with {bert_layers} layers")
            self.bert = BertEmbeddings.create_bert_embedder(
                model_name=bert_model_name,
                num_layers=bert_layers,
                freeze=freeze_bert
            )
            # BERT output size might be different from the model's d_model
            bert_output_dim = self.bert.config.hidden_size
            self.bert_projection = nn.Linear(bert_output_dim, d_model) if bert_output_dim != d_model else nn.Identity()
            
            # We don't need word_embedding when using BERT
            self.word_embedding = None
        else:
            self.bert = None
            # Core components with pre-trained embeddings support
            if pretrained_embeddings is not None:
                self.word_embedding = nn.Embedding.from_pretrained(
                    pretrained_embeddings,
                    freeze=False,  # Allow fine-tuning
                    padding_idx=0
                )
                logger.info(f"Initialized embedding layer with pretrained weights of shape {pretrained_embeddings.shape}")
            else:
                self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
                logger.info(f"Initialized random embedding layer of shape ({vocab_size}, {d_model})")
            
        # MLM pre-training head
        if enable_mlm:
            logger.info(f"Enabling MLM pre-training with probability {mlm_probability}")
            self.mlm_head = nn.Linear(d_model, vocab_size)
            
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout * 2)  # Higher dropout for input
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Feature extraction and processing
        self.feature_processor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in [3, 5, 7]
        ])
        
        # Transformer encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
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
            dropout=dropout,
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
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model, num_labels)
        )
        
        # Initialize model weights
        self.apply(self._init_weights)

    def mask_tokens(self, inputs, attention_mask, tokenizer):
        """
        Prepare masked tokens for MLM pre-training, similar to BERT
        
        Args:
            inputs: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            tokenizer: Tokenizer for special token IDs
        
        Returns:
            Tuple of masked inputs and labels for MLM
        """
        device = inputs.device
        labels = inputs.clone()
        
        # We only mask tokens with attention_mask = 1
        masked_indices = torch.bernoulli(torch.full(labels.shape, self.mlm_probability, device=device)).bool() & attention_mask.bool()
        
        # Don't mask special tokens like [CLS], [SEP], etc.
        special_tokens_mask = torch.tensor(
            [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool,
            device=device
        )
        masked_indices.masked_fill_(special_tokens_mask, False)
        
        # For masked tokens, set label to -100 to ignore in loss computation
        labels[~masked_indices] = -100
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%), keep the masked input tokens unchanged
        return inputs, labels

    def forward_mlm(self, input_ids, attention_mask, masked_lm_labels=None, tokenizer=None):
        """
        Forward pass for MLM pre-training
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            masked_lm_labels: Labels for masked LM (optional)
            tokenizer: Tokenizer for masking tokens (needed if masked_lm_labels is None)
            
        Returns:
            MLM loss
        """
        if masked_lm_labels is None and tokenizer is not None:
            # Create masked inputs and labels during training
            input_ids, masked_lm_labels = self.mask_tokens(input_ids.clone(), attention_mask, tokenizer)
        
        # Get contextualized embeddings (same as in the main forward method)
        if self.use_bert:
            # Use BERT embeddings
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embedded = outputs.last_hidden_state
            embedded = self.bert_projection(embedded)
        else:
            # Use standard embeddings + positional encoding
            embedded = self.word_embedding(input_ids)
            embedded = self.pos_encoder(embedded)
        
        # Apply transformer encoder
        embedded = self.embedding_norm(embedded)
        embedded = self.input_dropout(embedded)
        
        # Create a boolean mask from the attention mask (1 = attend, 0 = ignore)
        mask = attention_mask.bool()
        
        # Pass through transformer encoder
        transformer_outputs = self.transformer_encoder(embedded, src_key_padding_mask=~mask)
        
        # MLM prediction head
        prediction_scores = self.mlm_head(transformer_outputs)
        
        # Calculate MLM loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        mlm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        return mlm_loss

    def forward(self, input_ids, attention_mask, labels=None):
        # Initial embeddings
        if self.use_bert:
            # Use BERT embeddings
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            embedded = outputs.last_hidden_state
            embedded = self.bert_projection(embedded)
        else:
            # Use standard embeddings + positional encoding
            embedded = self.word_embedding(input_ids)
            embedded = self.pos_encoder(embedded)
        
        # Store original input for residual connections
        input_embedding = embedded
        
        # Feature extraction
        features = []
        x_conv = embedded.transpose(1, 2)
        for conv in self.feature_processor:
            features.append(conv(x_conv).transpose(1, 2))
        
        # Combine features
        x = sum(features) + input_embedding
        x = self.embedding_norm(x)
        
        # Create a boolean mask from the attention mask (1 = attend, 0 = ignore)
        mask = attention_mask.bool()
        batch_size = mask.size(0)
        
        # Pass through transformer encoder
        encoder_out = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Add residual connection
        encoder_out = self.residual_norm1(encoder_out + x)
        
        # BiLSTM for sequential modeling
        lstm_out, _ = self.bilstm(encoder_out)
        
        # Residual connection
        x = self.residual_norm2(lstm_out + encoder_out)
        
        # Pass through transformer decoder (using encoder outputs as memory)
        decoder_out = self.transformer_decoder(x, encoder_out, tgt_key_padding_mask=~mask, memory_key_padding_mask=~mask)
        
        # Cross attention between transformer and LSTM outputs
        cross_attn_out, _ = self.cross_attention(decoder_out, lstm_out, lstm_out, mask=mask)
        
        # Concatenate outputs for rich representation
        concat_output = torch.cat([decoder_out, cross_attn_out], dim=-1)
        final_repr = self.final_norm(concat_output)
        
        # Classification layer
        logits = self.classifier(final_repr)
        
        # Use standard cross entropy
        if labels is not None:
            # Training and validation loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(-100).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            # Return predictions (argmax)
            return logits.argmax(dim=-1)

    def _init_weights(self, module):
        """Initialize the weights of modules"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

def train_custom_model(train_loader, val_loader, tokenizer, epochs=50, lr=2e-5, 
                   warmup_steps=100, weight_decay=0.02, dropout=0.3,
                   gradient_accumulation_steps=2, d_model=768,
                   num_heads=12, num_layers=8, use_bert=True,
                   bert_model_name="bert-base-multilingual-cased", 
                   bert_layers=4, freeze_bert=False,
                   mlm_pretraining=True, mlm_epochs=5, mlm_probability=0.15):
    """
    Train a custom transformer model for idiom detection with BERT embeddings
    and optional MLM pre-training.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        tokenizer: Tokenizer for processing text
        epochs: Number of training epochs
        lr: Learning rate
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        dropout: Dropout rate
        gradient_accumulation_steps: Number of steps to accumulate gradients
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        use_bert: Whether to use BERT embeddings
        bert_model_name: Name of BERT model to use
        bert_layers: Number of BERT layers to use
        freeze_bert: Whether to freeze BERT parameters
        mlm_pretraining: Whether to use MLM pre-training
        mlm_epochs: Number of MLM pre-training epochs
        mlm_probability: Probability of masking tokens for MLM
    
    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model with updated architecture
    model = CustomTransformerForIdiomDetection(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_bert=use_bert,
        bert_model_name=bert_model_name,
        bert_layers=bert_layers,
        freeze_bert=freeze_bert,
        enable_mlm=mlm_pretraining,
        mlm_probability=mlm_probability
    ).to(device)
    
    logger.info(f"Model initialized with {'BERT embeddings' if use_bert else 'custom embeddings'}")
    logger.info(f"Model will {'use' if mlm_pretraining else 'not use'} MLM pre-training")
    
    # MLM Pre-training phase
    if mlm_pretraining and mlm_epochs > 0:
        logger.info(f"Starting MLM pre-training for {mlm_epochs} epochs")
        
        # Optimizer for pre-training
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * mlm_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            total_steps=total_steps
        )
        
        # Pre-training loop
        best_mlm_loss = float('inf')
        for epoch in range(mlm_epochs):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"MLM Pre-training Epoch {epoch+1}/{mlm_epochs}")
            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass with MLM
                mlm_loss = model.forward_mlm(input_ids, attention_mask, tokenizer=tokenizer)
                
                # Backward pass
                if gradient_accumulation_steps > 1:
                    mlm_loss = mlm_loss / gradient_accumulation_steps
                    
                mlm_loss.backward()
                
                # Update weights after gradient accumulation
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                total_loss += mlm_loss.item() * gradient_accumulation_steps
                avg_loss = total_loss / (i + 1)
                progress_bar.set_postfix(loss=avg_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    # Forward pass with MLM, create masked inputs for validation too
                    masked_input_ids, masked_lm_labels = model.mask_tokens(input_ids.clone(), attention_mask, tokenizer)
                    batch_loss = model.forward_mlm(masked_input_ids, attention_mask, masked_lm_labels=masked_lm_labels)
                    val_loss += batch_loss.item()
            
            # Calculate metrics
            val_loss /= len(val_loader)
            
            logger.info(f"MLM Pre-training Epoch {epoch+1}/{mlm_epochs}, Val Loss: {val_loss:.4f}")
            
            # Save best model based on MLM loss
            if val_loss < best_mlm_loss:
                best_mlm_loss = val_loss
                torch.save(model.state_dict(), 'best_mlm_model.pt')
                logger.info(f"New best MLM validation loss: {best_mlm_loss:.4f}, model saved")
        
        # Load the best MLM pre-trained model
        model.load_state_dict(torch.load('best_mlm_model.pt'))
        logger.info("MLM pre-training completed")

    # Fine-tuning phase
    # Optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Learning rate scheduler with linear warmup and decay
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0.0
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            loss = model(input_ids, attention_mask, labels)
            
            # Backward pass
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
                
            loss.backward()
            
            # Update weights after gradient accumulation
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * gradient_accumulation_steps
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix(loss=avg_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                loss = model(input_ids, attention_mask, labels)
                val_loss += loss.item()
                
                # Get predictions for metrics calculation
                predictions = model(input_ids, attention_mask)
                
                # Collect predictions and labels where attention mask is active
                for i, mask in enumerate(attention_mask):
                    pred = predictions[i][mask.bool()].cpu().tolist()
                    true = labels[i][mask.bool()].cpu().tolist()
                    val_preds.extend(pred)
                    val_labels.extend(true)
        
        # Calculate F1 score for idiom tokens (non-zero)
        val_loss /= len(val_loader)
        val_f1 = calculate_f1(val_preds, val_labels)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
            logger.info(f"New best F1: {best_val_f1:.4f}, model saved")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    logger.info(f"Training completed. Best F1: {best_val_f1:.4f}")
    return model

def calculate_f1(preds, labels):
    """Calculate F1 score for idiom detection (label 1 and 2)"""
    if not preds or not labels:
        return 0.0
    
    # Convert to numpy arrays if they're lists
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
        
    # Count true positives, false positives, false negatives
    tp = np.sum((preds != 0) & (labels != 0))
    fp = np.sum((preds != 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels != 0))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1 