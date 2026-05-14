import numpy as np
import pytorch_lightning as pl

import torch
import torch.nn as nn

from scipy.stats import pearsonr

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModel,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


class PatentSimilarityModule(pl.LightningModule):

    def __init__(
        self,
        model_name,
        token,
        lr,
        weight_decay,
        warmup_ratio,
        use_bf16,
        lora,
    ):
        super().__init__()

        self.save_hyperparameters()

        dtype = (torch.bfloat16 if use_bf16 else torch.float16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            token=token,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True,
            dtype=dtype,
        )

        self.backbone = prepare_model_for_kbit_training(
            self.backbone
        )

        self.backbone.config.use_cache = False

        lora_config = LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        hidden_size = (self.backbone.config.hidden_size)

        self.regressor = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(
                hidden_size // 2,
                1,
            ),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        stacked = torch.stack([
            hidden_states[-1],
            hidden_states[-2],
            hidden_states[-3],
            hidden_states[-4],
        ]).mean(0)

        seq_lengths = (attention_mask.sum(dim=1) - 1)

        pooled = stacked[
            torch.arange(
                stacked.size(0),
                device=stacked.device
            ),
            seq_lengths
        ]

        logits = self.regressor(pooled).squeeze(-1)

        return logits

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
        )

        loss = self.criterion(
            logits,
            batch["labels"],
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["labels"].size(0),
        )

        return loss

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.val_losses = []

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.criterion(logits, batch["labels"])
        preds = torch.sigmoid(logits)
        preds = preds.clamp(0, 1)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(batch["labels"].detach().cpu())
        self.val_losses.append(loss.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).float().numpy()
        labels = torch.cat(self.val_labels).float().numpy()
        loss = torch.stack(self.val_losses).mean()

        pearson = pearsonr(preds, labels)[0]

        mse = np.mean((preds - labels) ** 2)

        self.log_dict(
            {
                "val_loss": loss,
                "pearson": pearson,
                "mse": mse,
            },
            prog_bar=True,
            sync_dist=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = (self.trainer.estimated_stepping_batches)

        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = (
            get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }