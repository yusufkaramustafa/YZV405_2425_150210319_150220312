from dataset import get_dataloaders, debug_data_loader
from model import BertForIdiomDetection, train_model, evaluate, predict_idioms
import torch

train_loader, val_loader, tokenizer = get_dataloaders()

model = train_model(train_loader, val_loader, epochs=5, lr=2e-5)

    