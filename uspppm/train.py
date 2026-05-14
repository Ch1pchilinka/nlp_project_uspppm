import os
import hydra
import pytorch_lightning as pl

from dotenv import load_dotenv

from omegaconf import DictConfig

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)

from data import PatentDataModule
from model import PatentSimilarityModule
from utils import clean_mem
from logger import get_logger


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    load_dotenv()
    clean_mem()

    pl.seed_everything(cfg.train.seed)

    datamodule = PatentDataModule(
        train_csv=cfg.data.train_csv,
        cpc_csv=cfg.data.cpc_csv,
        model_name=cfg.model.model_name,
        token=os.getenv(cfg.secret.token),
        max_len=cfg.train.max_len,
        train_batch_size=cfg.train.train_batch_size,
        valid_batch_size=cfg.train.valid_batch_size,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
    )

    model = PatentSimilarityModule(
        model_name=cfg.model.model_name,
        token=os.getenv(cfg.secret.token),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        use_bf16=cfg.train.use_bf16,
        lora=cfg.train.lora,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="artifacts/checkpoints",
            filename="best",
            monitor="pearson",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="pearson",
            patience=2,
            mode="max",
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        accumulate_grad_batches=(cfg.train.accumulate_grad_batches),
        gradient_clip_val=(cfg.train.gradient_clip_val),
        log_every_n_steps=(cfg.train.log_every_n_steps),
        callbacks=callbacks,
        default_root_dir=cfg.train.output_dir,
        logger=get_logger(cfg),
        num_sanity_val_steps=cfg.train.num_sanity_val_steps
    )

    trainer.fit(
        model,
        datamodule=datamodule
    )

    


if __name__ == "__main__":
    train()