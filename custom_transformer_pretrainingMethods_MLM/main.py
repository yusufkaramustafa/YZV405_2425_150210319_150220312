from dataset import get_dataloaders, debug_data_loader
from custom_transformer import CustomTransformerForIdiomDetection, train_custom_model
from transformers import BertTokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU setup
logger.info(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")

# Use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
logger.info("Tokenizer loaded successfully")

# Get dataloaders with BERT tokenizer
train_loader, val_loader = get_dataloaders(tokenizer=tokenizer, batch_size=16)
logger.info("Dataloaders created successfully")

# Debug first batch to verify data format
logger.info("Debugging first batch:")
debug_data_loader(train_loader, tokenizer)


logger.info("Starting model training...")
model = train_custom_model(
    train_loader=train_loader,
    val_loader=val_loader,
    tokenizer=tokenizer,
    epochs=30, 
    lr=3e-5,
    # BERT configuration 
    use_bert=True,
    bert_model_name="bert-base-multilingual-cased",
    bert_layers=4,
    freeze_bert=False,
    # MLM pre-training configuration
    mlm_pretraining=True,
    mlm_epochs=5,
    mlm_probability=0.15,
    # Model configuration
    d_model=768,
    num_heads=12,
    num_layers=8,
    dropout=0.2,
    gradient_accumulation_steps=2
)

logger.info("Model training completed!")

# Save the final model
torch.save(model.state_dict(), 'final_model.pt')
logger.info("Final model saved to final_model.pt")

