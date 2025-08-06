from transformers import AutoTokenizer
from collections import Counter
import json

def compare(model1, model2):
    print(f"Downloading tokenizer for {model1}")
    tokenizer1 = AutoTokenizer.from_pretrained(model1)
    
    print(f"Downloading tokenizer for {model2}")
    tokenizer2 = AutoTokenizer.from_pretrained(model2)
    
    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    
    print(f"\n===RESULTS===")
    print(f"{model1}:")
    print(f"Vocab len:{len(vocab1)}")
    
    print(f"{model2}:")
    print(f"Vocab len:{len(vocab2)}")

    vocab1_keys = set(vocab1.keys())
    vocab2_keys = set(vocab2.keys())
    
    common_tokens = vocab1_keys.intersection(vocab2_keys)
    only_in_model1 = vocab1_keys.difference(vocab2_keys)
    only_in_model2 = vocab2_keys.difference(vocab1_keys)
    
    print(f"Common tokens:{len(common_tokens)}")
    print(f"Unique in model{model1}:{len(only_in_model1)}")
    print(f"Unique in model{model2}:{len(only_in_model2)}")
    
    special_tokens1 = set()
    special_tokens2 = set()
    
    for attr in ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']:
        token1 = getattr(tokenizer1, attr, None)
        token2 = getattr(tokenizer2, attr, None)
        if token1:
            special_tokens1.add(token1)
        if token2:
            special_tokens2.add(token2)
    
    print(f"\n===SPECIAL TOKENS===")
    print(f"{model1}: {special_tokens1}")
    print(f"{model2}: {special_tokens2}")
    
    print(f"\n===ANALYSIS===")
    print(f"\n===COMPATIBILITY CHECK===")
    if len(common_tokens) > len(vocab1) * 0.8:
        print("High dictionary compatibility (>80% common tokens)")
    elif len(common_tokens) > len(vocab1) * 0.5:
        print("Average dictionary compatibility (50-80% common tokens)")
    else:
        print("Low dictionary compatibility (<50% common tokens)")

    results = {
        'model1': {
            'name': model1,
            'vocab_size': len(vocab1),
            'special_tokens': list(special_tokens1)
        },
        'model2': {
            'name': model2,
            'vocab_size': len(vocab2),
            'special_tokens': list(special_tokens2)
        },
        'comparison': {
            'common_tokens': len(common_tokens),
            'only_in_model1': len(only_in_model1),
            'only_in_model2': len(only_in_model2),
            'sample_common_tokens': list(list(common_tokens)[:20])
        }
    }
    
    with open('vocab_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results are saved to file 'vocab_comparison.json'")
    
    return {
        'tokenizer1': tokenizer1,
        'tokenizer2': tokenizer2,
        'vocab1': vocab1,
        'vocab2': vocab2,
        'statistics': results
    }
    
def detailed_analysis(result):
    vocab1 = result['vocab1']
    vocab2 = result['vocab2']
    
    print(f"\n=== DETAILED ANALYSIS ===")

    lengths1 = [len(token) for token in vocab1.keys()]
    lengths2 = [len(token) for token in vocab2.keys()]
    
    print(f"Average token length in {result['statistics']['model1']['name']}: {sum(lengths1)/len(lengths1):.2f}")
    print(f"Average token length in {result['statistics']['model2']['name']}: {sum(lengths2)/len(lengths2):.2f}")
    
    prefixes1 = Counter([token[:2] for token in vocab1.keys() if len(token) > 2])
    prefixes2 = Counter([token[:2] for token in vocab2.keys() if len(token) > 2])
    
    print(f"\nPopular prefixes in {result['statistics']['model1']['name'][:20]}:")
    for prefix, count in prefixes1.most_common(10):
        print(f"  '{prefix}': {count}")
    
    print(f"\nPopular prefixes in {result['statistics']['model2']['name'][:20]}:")
    for prefix, count in prefixes2.most_common(10):
        print(f"  '{prefix}': {count}")

if __name__ == "__main__":
    model1 = input("Enter model#1 name. For example 'unsloth/Llama-3.2-1B-Instruct':")
    model2 = input("Enter model#1 name. For example 'unsloth/Llama-3.2-1B-Instruct':")
    
    try:
        comparison_result = compare(model1, model2)
        
        detailed_analysis(comparison_result)
    except Exception as e:
        print(f"An error occured while comparing tokenizers:{e}")      