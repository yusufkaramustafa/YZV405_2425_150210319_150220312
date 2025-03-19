from dataset import get_dataloaders, debug_data_loader

# Get DataLoaders
train_loader, val_loader, tokenizer = get_dataloaders()

# Debug label alignment
debug_data_loader(train_loader, tokenizer)