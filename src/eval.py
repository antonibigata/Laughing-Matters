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

from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils
from src.utils.torch_utils import load_checkpoint
from src.utils.utils import configure_cfg_from_checkpoint, save_summary

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    cfg = configure_cfg_from_checkpoint(cfg)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    video_rate = 25
    if hasattr(datamodule, "video_rate"):
        video_rate = datamodule.video_rate
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        sample_rate=cfg.datamodule.get("audio_rate"),
        video_rate=video_rate,
        eval_save_all=cfg.eval_save_all,
        compile_model=cfg.compile_model,
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    save_summary(model, cfg.paths.output_dir, logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # net = hydra.utils.instantiate(cfg.model.net)
    # net.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
    # model.load_from_checkpoint(cfg.ckpt_path, strict=False, net=net)
    if cfg.load_separetely:
        net = hydra.utils.instantiate(cfg.model.net)

        net = load_checkpoint(
            net,
            cfg.ckpt_path,
            allow_extra_keys=True,
            extra_key="state_dict",
            replace=("model.", ""),
            map_location="cuda",
        )
        # cfg.model.num_frames_val = default(cfg.n_generate_frames, cfg.model.num_frames_val)
        video_rate = 25
        if hasattr(datamodule, "video_rate"):
            video_rate = datamodule.video_rate

        # model.load_from_checkpoint(cfg.ckpt_path, net=net)
        model.update_inf_model(net.to("cuda"))
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
