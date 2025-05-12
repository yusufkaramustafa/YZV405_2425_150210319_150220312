import torch
import argparse
import os
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from dataset import get_dataloaders
from model import IdiomDetectionModel, evaluate
from torch.cuda.amp import autocast, GradScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Train idiom detection model")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", 
                        help="Model to use (default: bert-base-multilingual-cased)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=128, 
                        help="Maximum sequence length (default: 128)")
    parser.add_argument("--train_path", type=str, default="data/train.csv", 
                        help="Path to training data")
    parser.add_argument("--val_path", type=str, default="data/eval.csv", 
                        help="Path to validation data")
    parser.add_argument("--save_dir", type=str, default="saved_models", 
                        help="Directory to save model")
    parser.add_argument("--fp16", action="store_true", 
                        help="Enable mixed precision training")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.fp16 and device.type == 'cuda' else None
    if args.fp16 and device.type == 'cuda':
        print("Mixed precision training enabled")
    elif args.fp16 and device.type != 'cuda':
        print("Warning: Mixed precision requested but CUDA is not available. Using full precision.")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, tokenizer = get_dataloaders(
        train_path=args.train_path,
        val_path=args.val_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_name=args.model_name
    )
    
    # Initialize model
    print(f"Initializing {args.model_name} model...")
    model = IdiomDetectionModel(
        model_name=args.model_name, 
        num_labels=3  # O, B-IDIOM, I-IDIOM
    ).to(device)
    
    # Optimizer with differential learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # BERT parameters
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' in n],
            'weight_decay': 0.01,
            'lr': args.lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' in n],
            'weight_decay': 0.0,
            'lr': args.lr
        },
        # Custom layers - higher learning rate
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' not in n],
            'weight_decay': 0.01,
            'lr': args.lr * 5  # Higher learning rate for custom layers
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' not in n],
            'weight_decay': 0.0,
            'lr': args.lr * 5  # Higher learning rate for custom layers
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    from tqdm import tqdm
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            
            # Forward pass
            optimizer.zero_grad()
            
            if args.fp16 and device.type == 'cuda':
                # Mixed precision forward pass
                with autocast():
                    outputs = model(**batch)
                    loss = outputs['loss']
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard full precision pass
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation - use the standard evaluate function from model.py
        print("Evaluating...")
        metrics = evaluate(model, val_loader, tokenizer, device)
        
        print(f"Epoch {epoch+1} results:")
        print(f"  Training loss: {avg_train_loss:.4f}")
        print(f"  Validation loss: {metrics['loss']:.4f}")
        print(f"  F1 score: {metrics['f1']:.4f}")
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_path = os.path.join(args.save_dir, f"{args.model_name.replace('/', '-')}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}!")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"{args.model_name.replace('/', '-')}_checkpoint.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'scaler': scaler.state_dict() if scaler else None,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
