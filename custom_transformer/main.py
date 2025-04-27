from dataset import get_dataloaders, debug_data_loader
from custom_transformer import CustomTransformerForIdiomDetection, train_custom_model
from transformers import BertTokenizer
import torch

# GPU setup
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# Use BERT tokenizer (only for tokenization, not weights)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
print("\nTokenizer loaded successfully")

# Get dataloaders with BERT tokenizer
train_loader, val_loader = get_dataloaders(tokenizer=tokenizer, batch_size=16)
print("\nDataloaders created successfully")

# Debug first batch to verify data format
print("\nDebugging first batch:")
debug_data_loader(train_loader, tokenizer)


print("\nStarting model training...")
model = train_custom_model(
    train_loader=train_loader,
    val_loader=val_loader,
    tokenizer=tokenizer,
    epochs=30, 
    lr=3e-5  
)

