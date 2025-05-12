import pandas as pd
import torch
import argparse
import os
from transformers import AutoTokenizer
from model import IdiomDetectionModel, predict_idioms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a single model on all data")
    parser.add_argument("--test_path", type=str, required=True, 
                        help="Path to test CSV file")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save output CSV file")
    parser.add_argument("--model_dir", type=str, required=True, 
                        help="Directory containing saved models")
    parser.add_argument("--model_name", type=str, default="google-bert-bert-base-multilingual-cased_best.pt", 
                        help="Model filename to use")
    parser.add_argument("--model_type", type=str, default="google-bert/bert-base-multilingual-cased", 
                        help="Base model type")
    parser.add_argument("--max_length", type=int, default=128, 
                        help="Maximum sequence length")
    return parser.parse_args()

def load_model(model_path, base_model_name, device):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Create model with right bert base
    model = IdiomDetectionModel(model_name=base_model_name, num_labels=3)
    
    # Load weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer

def run_inference():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_path}")
    test_df = pd.read_csv(args.test_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Initialize model and tokenizer
    print(f"Loading model: {args.model_name}")
    
    # Path to saved model
    model_path = os.path.join(args.model_dir, args.model_name)
    
    # Load model
    model, tokenizer = load_model(model_path, args.model_type, device)
    
    # Process test data
    print("Running inference on all test data...")
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        id_val = row['id']
        lang = row['language']
        sentence = row['sentence']
        
        # Get prediction - call the module function instead of a class method
        results_tuple = predict_idioms(model, tokenizer, sentence, device)
        
        # Unpack results and idiom indices
        _, idiom_indices = results_tuple
        
        # Format result: If no idioms detected, use [-1]
        if not idiom_indices:
            formatted_indices = [-1]
        else:
            # Flatten the list of lists if needed
            formatted_indices = []
            for indices in idiom_indices:
                if isinstance(indices, list):
                    formatted_indices.extend(indices)
                else:
                    formatted_indices.append(indices)
            
            # Sort indices
            formatted_indices = sorted(formatted_indices)
        
        results.append({
            'id': id_val,
            'language': lang,
            'indices': formatted_indices
        })
    
    # Convert results to DataFrame and save
    output_df = pd.DataFrame(results)
    
    # Convert indices to string format
    output_df['indices'] = output_df['indices'].apply(lambda x: str(x))
    
    # Save to CSV
    output_df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    run_inference()