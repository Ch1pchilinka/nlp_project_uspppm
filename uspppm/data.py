import pandas as pd
import pytorch_lightning as pl
import torch

from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import Dataset, DataLoader

from transformers import DataCollatorWithPadding

from utils import (
    build_prompt,
    get_tokenizer,
)


class PatentDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_len,
    ):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                row["score"],
                dtype=torch.float
            )
        }


class PatentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv,
        cpc_csv,
        model_name,
        token,
        max_len,
        train_batch_size,
        valid_batch_size,
        num_workers,
        seed,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        df = pd.read_csv(
            self.hparams.train_csv
        )

        cpc_df = pd.read_csv(
            self.hparams.cpc_csv
        )

        cpc_map = dict(
            zip(cpc_df["code"], cpc_df["title"])
        )

        df["context_text"] = df[
            "context"
        ].map(
            lambda x: cpc_map.get(x, x)
        )

        df["text"] = df.apply(
            build_prompt,
            axis=1
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=self.hparams.seed,
        )

        train_idx, valid_idx = next(
            splitter.split(
                df,
                groups=df["anchor"]
            )
        )

        train_df = df.iloc[
            train_idx
        ].reset_index(drop=True)

        valid_df = df.iloc[
            valid_idx
        ].reset_index(drop=True)

        tokenizer = get_tokenizer(
            self.hparams.model_name,
            self.hparams.token,
        )

        self.collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )

        self.train_dataset = PatentDataset(
            train_df,
            tokenizer,
            self.hparams.max_len,
        )

        self.valid_dataset = PatentDataset(
            valid_df,
            tokenizer,
            self.hparams.max_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.valid_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )