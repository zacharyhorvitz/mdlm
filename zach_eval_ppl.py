# evaluate perplexity with perplexity, gpt2-xl



import json

from evaluate import load
import numpy as np



path = '/home/horvitz/twisty-diffusion/mdlm/outputs/openwebtext-train/2024.09.02/181542/sample_evaluation/20240902-182734/text_samples.jsonl'

with open(path, 'r') as f:
    texts = [json.loads(line)['text'] for line in f]


from transformers import AutoTokenizer

# load roberta tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

# trim texts to 50 roberta tokens


# and decode them back to strings
perp_model = "gpt2-xl"
# perp_model = "gpt2"

texts = [tokenizer.encode(text, max_length=50, truncation=True) for text in texts]
texts = [tokenizer.decode(text) for text in texts]

print(texts)

perplexity = load("perplexity", module_type="metric")
perps = perplexity.compute(predictions=texts, model_id=perp_model)

print(perps)

print(perps['mean_perplexity'])