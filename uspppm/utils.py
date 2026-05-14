import random
import numpy as np
import torch
import gc

from transformers import AutoTokenizer


def build_prompt(row):

    return f"""
Patent Classification:
{row['context_text']}

Anchor Phrase:
{row['anchor']}

Target Phrase:
{row['target']}

Task:
Predict semantic similarity score between the anchor phrase and target phrase.
""".strip()

def get_tokenizer(model_name: str, token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer

def clean_mem():
    for obj in dir():
        if 'model' in obj.lower() or 'train' in obj.lower() or 'valid' in obj.lower():
            del globals()[obj]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")