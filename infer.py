import dotenv
import hydra

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

import logging
from typing import List, Optional

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

import hydra
from omegaconf import DictConfig, OmegaConf

from celltrack.utils import utils
from celltrack.datamodules.celltrack_datamodule_mulSeq import CellTrackDataModule
from celltrack.models.celltrack_plmodel import CellTrackLitModel

log = logging.getLogger(__name__)

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

    # Init Lightning loggers
    logger: List[Logger] = []
    # logger.append(CSVLogger(**config.logger.csv))
    logger.append(TensorBoardLogger(**config.logger.tensorboard))

    # Init Lightning trainer
    trainer = Trainer(**config.trainer, logger=logger)

    # Send some parameters from config to all lightning loggers    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        logger=logger,
        callbacks=None
    )

    ########################################
    ########################################
    ##### model training and testing #######
    ########################################
    ########################################

    # Load from the checkpoint
    model = CellTrackLitModel.load_from_checkpoint(**config.inference)

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
        logger=logger,
        callbacks=None
    )

    # Print path to best checkpoint
    # log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

#  config_multiLabel config
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    filter_keys = list(config.keys())
    filter_keys.remove('callbacks')
    config = OmegaConf.masked_copy(config, keys=filter_keys)
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
