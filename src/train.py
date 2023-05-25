import pyrootutils
import os

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = ""

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
import torch

from src import utils
from src.utils.utils import save_summary
from src.utils.utils import get_latest_checkpoint_and_config

log = utils.get_pylogger(__name__)

try:
    import horovod.torch as hvd
except ImportError:
    log.warning("Horovod not installed! Distributed training will not work!")


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    if cfg.trainer.get("strategy") == "horovod":
        hvd.init()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    debug = cfg.trainer.get("fast_dev_run")
    video_rate = 25
    if hasattr(datamodule, "video_rate"):
        video_rate = datamodule.video_rate
    devices = cfg.trainer.get("devices", 1)
    if devices == "auto":
        devices = torch.cuda.device_count()
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        debug=debug,
        sample_rate=cfg.datamodule.get("audio_rate"),
        video_rate=video_rate,
        n_devices=devices,
    )

    log.info("Instantiating loggers...")
    if cfg.trainer.get("strategy") == "horovod":
        if hvd.rank() == 0:
            logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
        else:
            logger = None
    else:
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))
    if cfg.get("find_lr"):
        logger.wandb.name = None

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"), logger=logger)

    save_summary(model, cfg.paths.output_dir, logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("find_lr"):
        log.info("Finding learning rate...")
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print("lr_finder resuls: ", lr_finder.results)
        print("lr finder suggestion: ", lr_finder.suggestion())
        exit()

    if cfg.get("train"):
        log.info("Starting training!")
        checkpoint_path = cfg.get("ckpt_path")
        if checkpoint_path and os.path.isdir(checkpoint_path):
            checkpoint_path, _ = get_latest_checkpoint_and_config(cfg.get("ckpt_path"))
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
