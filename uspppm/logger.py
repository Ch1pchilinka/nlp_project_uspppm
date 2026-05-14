"""Resolve the configured logger from Hydra config."""

from omegaconf import DictConfig
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers import MLFlowLogger

def get_logger(cfg: DictConfig) -> Logger:
    params = cfg.logger.mlflow
    
    return MLFlowLogger(
        tracking_uri=params.tracking_uri,
        experiment_name=params.experiment_name,
        run_name=params.run_name,
    )