import dotenv
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

import logging
from typing import List, Optional

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

from omegaconf import OmegaConf, DictConfig

from celltrack.utils import utils
from celltrack.datamodules.celltrack_datamodule_mulSeq import CellTrackDataModule
from celltrack.models.celltrack_plmodel import CellTrackLitModel

log = logging.getLogger(__name__)

''' Remove hydra and pytorch lightning components
[x] convert hydra to simple omegaconf
[x] convert logging module to wandb
'''

def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    ########################################
    ########################################
    ######### lightning config #############
    ########################################
    ########################################

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    datamodule = CellTrackDataModule(**config.datamodule)

    # Init Lightning model
    model = CellTrackLitModel(**config.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    callbacks.append(ModelCheckpoint(**config.callbacks.model_checkpoint))
    callbacks.append(EarlyStopping(**config.callbacks.early_stopping))

    # Init Lightning loggers
    logger: List[Logger] = []
    # logger.append(TensorBoardLogger(**config.logger.tensorboard))
    logger.append(WandbLogger(**config.logger.wandb))
    
    # Init Lightning trainer
    trainer = Trainer(**config.trainer, callbacks=callbacks, logger=logger)

    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    ########################################
    ########################################
    ##### model training and testing #######
    ########################################
    ########################################

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule)

    ########################################
    ########################################
    ######### lightning config #############
    ########################################
    ########################################

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

def load_conf():
    cfg_cmd = OmegaConf.from_cli()  
    cfg = OmegaConf.load('configs/config.yaml')
    cfg = OmegaConf.merge(cfg, cfg_cmd)
    for cfg_path in cfg.defaults:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(os.path.join('configs', cfg_path)))
    return cfg

def mkdir(cfg:DictConfig):
    log_dir = cfg.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
if __name__ == "__main__":
    cfg = load_conf()
    mkdir(cfg)
    train(cfg)
