import random
import torch
import numpy as np
from transformers import AutoTokenizer


def build_prompt(row):
    result = f"""
            Patent Classification:
            {row['context_text']}

            Anchor Phrase:
            {row['anchor']}

            Target Phrase:
            {row['target']}

            Task:
            Predict semantic similarity score between the anchor phrase and target phrase.
            """.strip()
    return result


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_tokenizer(model_name, token):

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer