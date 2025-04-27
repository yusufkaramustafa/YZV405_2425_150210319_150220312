from transformers import BertTokenizer
import torch

def test_tokenization_and_mapping():
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Sample test cases
    test_cases = [
        {
            'sentence': "Pandemi döneminde olanların hesabını kimler verecek?",
            'idiom_indices': [3, 5],
            'tokenized_words': ['Pandemi', 'döneminde', 'olanların', 'hesabını', 'kimler', 'verecek', '?']
        },
        {
            'sentence': "E' un imperativo voltare la faccia a chi ti manca di rispetto",
            'idiom_indices': [4, 6],
            'tokenized_words': ['E', "'", 'un', 'imperativo', 'voltare', 'la', 'faccia', 'a', 'chi', 'ti', 'manca', 'di', 'rispetto']
        }
    ]
    
    for case in test_cases:
        print("\n" + "="*80)
        print(f"Testing sentence: {case['sentence']}")
        
        encoding = tokenizer.encode_plus(
            case['sentence'],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        print("\nTokenization Analysis:")
        print(f"{'Original Word':<15} {'BERT Tokens':<30} {'Word Index':<10}")
        print("-" * 60)
        
        word_idx = 0
        current_tokens = []
        
        for token in tokens:
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                if current_tokens:
                    print(f"{case['tokenized_words'][word_idx]:<15} {' '.join(current_tokens):<30} {word_idx:<10}")
                    current_tokens = []
                    word_idx += 1
                print(f"{'<special>':<15} {token:<30} {'N/A':<10}")
                continue
                
            if not token.startswith('##'):
                if current_tokens:
                    print(f"{case['tokenized_words'][word_idx]:<15} {' '.join(current_tokens):<30} {word_idx:<10}")
                    current_tokens = []
                    word_idx += 1
                current_tokens = [token]
            else:
                current_tokens.append(token)
        
        print("\nIdiom Analysis:")
        print(f"Idiom indices: {case['idiom_indices']}")
        print(f"Idiom words: {[case['tokenized_words'][i] for i in case['idiom_indices']]}")

def test_evaluation_metrics():
    print("\nTesting F1 Score Calculation:")
    print("=" * 50)
    
    test_cases = [
        {
            'pred': [3, 5],
            'truth': [3, 5],
            'expected_f1': 1.0,
            'desc': "Perfect match"
        },
        {
            'pred': [3],
            'truth': [3, 5],
            'expected_f1': 0.5,
            'desc': "Partial prediction"
        },
        {
            'pred': [3, 4, 5],
            'truth': [3, 5],
            'expected_f1': 0.8,
            'desc': "Extra prediction"
        },
        {
            'pred': [1, 2],
            'truth': [3, 5],
            'expected_f1': 0.0,
            'desc': "Completely wrong"
        },
        {
            'pred': [-1],
            'truth': [-1],
            'expected_f1': 1.0,
            'desc': "Both no idiom"
        }
    ]
    
    for case in test_cases:
        pred_set = set(case['pred'])
        truth_set = set(case['truth'])
        
        intersection = len(pred_set & truth_set)
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        recall = intersection / len(truth_set) if len(truth_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nTest Case: {case['desc']}")
        print(f"Predictions: {case['pred']}")
        print(f"Ground Truth: {case['truth']}")
        print(f"Calculated F1: {f1:.4f}")
        print(f"Expected F1: {case['expected_f1']:.4f}")
        print(f"Match: {'✓' if abs(f1 - case['expected_f1']) < 1e-6 else '✗'}")

def run_all_tests():
    print("Running tokenization and mapping tests...")
    test_tokenization_and_mapping()
    
    print("\nRunning evaluation metric tests...")
    test_evaluation_metrics()

if __name__ == "__main__":
    run_all_tests()
