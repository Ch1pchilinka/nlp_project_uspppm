import torch
import pandas as pd

from utils import build_prompt

import glob
import os
import hydra
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer
from model import PatentSimilarityModule
from omegaconf import DictConfig
from utils import clean_mem
from dotenv import load_dotenv
from tqdm import tqdm

cpc_df = pd.read_csv("./data/archive/titles.csv")
test_df = pd.read_csv("./data/test.csv")
cpc_map = dict(zip(cpc_df["code"],cpc_df["title"]))

def enrich_context(cpc):
    return cpc_map.get(cpc, cpc)

def predict_similarity(
    model,
    tokenizer,
    context,
    anchor,
    target,
    max_len,
):

    row = {
        "context_text": enrich_context(
            context
        ),
        "anchor": anchor,
        "target": target,
    }

    text = build_prompt(row)

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    device = model.device

    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device))
        score = torch.sigmoid(logits).item()

    return float(max(0.0, min(1.0, score)))

@hydra.main(config_path="../conf", config_name="config", version_base=None)

def infer(cfg: DictConfig):
    load_dotenv()
    clean_mem()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatentSimilarityModule.load_from_checkpoint(
        cfg.inference.checkpoint,
        map_location="cpu",
        weights_only=False,
        strict=False
    )

    model = model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model.hparams.model_name,
        token=os.getenv(cfg.secret.token)
    )

    scores = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        score = predict_similarity(
            model=model,
            tokenizer=tokenizer,
            context=row["context"],
            anchor=row["anchor"],
            target=row["target"],
            max_len=cfg.train.max_len,
        )
        scores.append(score)

    scores = np.clip(scores, 0, 1)

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "score": scores
    })

    submission_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    infer()