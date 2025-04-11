from dataset import get_dataloaders, debug_data_loader
from model import BertForIdiomDetection, train_model, evaluate, predict_idioms
import torch

train_loader, val_loader, tokenizer = get_dataloaders()

model = train_model(train_loader, val_loader, epochs=5, lr=2e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = predict_idioms(model, tokenizer, "Your test sentence here", device)

print("\nIdiom Detection Results:")
for token, pred in results:
    label = "IDIOM" if pred == 1 else "NORMAL"
    print(f"{token}: {label}")
    