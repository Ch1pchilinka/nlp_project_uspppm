import torch
import pandas as pd

def predict_similarity(
    model,
    tokenizer,
    context,
    anchor,
    target,
    max_len,
    device
):

    model.eval()

    context = enrich_context(context)

    text = f"""
Patent Classification:
{context}

Anchor Phrase:
{anchor}

Target Phrase:
{target}

Task:
Predict semantic similarity score between the anchor phrase and target phrase.
""".strip()

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        score = torch.sigmoid(
            logits
        ).item()

    score = max(0.0, min(1.0, score))

    return score

cpc_df = pd.read_csv("./data/archive/titles.csv")
cpc_map = dict(zip(cpc_df["code"], cpc_df["title"]))
def enrich_context(cpc):
    return cpc_map.get(cpc, cpc)