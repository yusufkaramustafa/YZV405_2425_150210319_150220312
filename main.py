import torch
from transformers import BertTokenizer

sentence = "İtibarlı bir kimse için başkasına elini mecburen açmak, ölmekten daha zordur."

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

tokens = tokenizer.tokenize(sentence)

print("Tokens from base cased tokenizer \n ", tokens, "\n")

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
tokens = tokenizer.tokenize(sentence)

print("Tokens from base uncased tokenizer \n ", tokens, "\n")

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", strip_accents = True)
tokens = tokenizer.tokenize(sentence)

print("Tokens from strip accents cased tokenizer \n ", tokens, "\n")

