import os
import gc
import warnings
import hydra

warnings.filterwarnings("ignore")

import numpy as np

from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import (
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from utils import get_tokenizer, seed_everything
from data import get_train_val_df, PatentDataset
from model import PatentSimilarityModel
from omegaconf import DictConfig
from dotenv import load_dotenv

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_CSV = "./data/train.csv"
OUTPUT_DIR = "./llama_patent_similarity"
MAX_LEN = 384
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_bf16_supported()
load_dotenv()
TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def train_epoch(model, loader, optimizer, criterion, scheduler):

    model.train()

    total_loss = 0

    optimizer.zero_grad()

    progress_bar = tqdm(loader)

    for step, batch in enumerate(progress_bar):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(
            logits,
            labels
        )

        loss = loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item()

        progress_bar.set_description(
            f"train_loss={loss.item():.4f}"
        )

    return total_loss / len(loader)

def valid_epoch(model, loader, criterion):

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        progress_bar = tqdm(loader)

        for batch in progress_bar:

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels
            )

            preds = torch.sigmoid(logits)

            preds = preds.clamp(0, 1)

            total_loss += loss.item()

            all_preds.extend(
                preds.detach().cpu().numpy()
            )

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

    pearson = pearsonr(
        all_preds,
        all_labels
    )[0]

    mse = np.mean(
        (
            np.array(all_preds)
            - np.array(all_labels)
        ) ** 2
    )

    return (
        total_loss / len(loader),
        pearson,
        mse
    )


def main():
    seed_everything(SEED)
    
    train_df, valid_df = get_train_val_df(seed=42, train_csv=TRAIN_CSV)
    tokenizer = get_tokenizer(MODEL_NAME, TOKEN)
    train_dataset = PatentDataset(
        dataframe=train_df,
        max_len=MAX_LEN,
        tokenizer=tokenizer
    )
    valid_dataset = PatentDataset(
        dataframe=valid_df,
        max_len=MAX_LEN,
        tokenizer=tokenizer
    )
    collator = DataCollatorWithPadding(
    tokenizer=train_dataset.tokenizer,
    pad_to_multiple_of=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    model = PatentSimilarityModel(MODEL_NAME, TOKEN, USE_BF16)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = (
        len(train_loader)
        * EPOCHS
        // GRAD_ACCUM_STEPS
    )
    warmup_steps = int(
        total_steps * WARMUP_RATIO
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )


    best_pearson = -1
    pearson = -1
    for epoch in range(EPOCHS):

        print(f"\n======== EPOCH {epoch+1}/{EPOCHS} ========")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler
        )

        valid_loss, pearson, mse = valid_epoch(
            model,
            valid_loader,
            criterion=criterion
        )

        print(f"\nTrain Loss : {train_loss:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Pearson    : {pearson:.4f}")
        print(f"MSE        : {mse:.4f}")

        if pearson > best_pearson:

            best_pearson = pearson

            os.makedirs(
                OUTPUT_DIR,
                exist_ok=True
            )

            model.backbone.save_pretrained(
                OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                OUTPUT_DIR
            )

            torch.save(
                model.state_dict(),
                f"{OUTPUT_DIR}/model.pt"
            )

            print("\nBest model saved!")

        gc.collect()

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()