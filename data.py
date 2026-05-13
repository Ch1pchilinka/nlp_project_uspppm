import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

import torch

from torch.utils.data import Dataset
from utils import build_prompt
from transformers import AutoTokenizer


def get_train_val_df(seed, train_csv):
    df = pd.read_csv(train_csv)

    cpc_df = pd.read_csv("./data/archive/titles.csv")
    cpc_map = dict(zip(cpc_df["code"], cpc_df["title"]))
    def enrich_context(cpc):
        return cpc_map.get(cpc, cpc)

    df["context_text"] = df["context"].apply(enrich_context)
    df["text"] = df.apply(build_prompt, axis=1)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=seed,
    )

    train_idx, valid_idx = next(
        splitter.split(
            df,
            groups=df["anchor"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    return train_df, valid_df


class PatentDataset(Dataset):

    def __init__(
        self,
        dataframe,
        max_len,
        tokenizer,
    ):

        self.df = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = row["text"]
        score = row["score"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(score, dtype=torch.float)
        }
