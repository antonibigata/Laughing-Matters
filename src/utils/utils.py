import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List
from functools import wraps
import os
import torch
from einops import repeat
import wandb
import hydra
import yaml
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import model_summary
from einops import rearrange
from src.utils import pylogger, rich_utils
import torchvision
import torchaudio
import moviepy.editor as mpy

log = pylogger.get_pylogger(__name__)


def identity(t, *args, **kwargs):
    return t


def first(arr, d=None):
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def cast_tuple(val, length=None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig, logger=None) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for cb_key, cb_conf in callbacks_cfg.items():

        if (logger is None or not len(logger)) and cb_key == "learning_rate_monitor":
            continue
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


@rank_zero_only
def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))
            try:
                wandb.run.log_code(".")  # log code to wandb
            except AttributeError:
                print("Wandb logger is not used")

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def unflatten_wandb_config(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split("/")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        if isinstance(value, dict):
            value = value["value"]
        d[parts[-1]] = value
    return resultDict


def get_latest_checkpoint(folder_path, extra_cond=None):
    """Returns path to the latest checkpoint in the folder."""

    if not os.path.exists(folder_path):
        raise Exception(f"Folder not found! <folder_path={folder_path}>")

    for root, _, files in os.walk(folder_path):
        if extra_cond is not None and extra_cond.split("-")[-1] not in root:
            continue
        checkpoints = []
        for file in files:
            if file.endswith(".ckpt"):
                checkpoints += [os.path.join(root, file)]
                if "last" in file:
                    return os.path.join(root, file)
        # checkpoints = [os.path.join(root, file) for file in files if file.endswith(".ckpt")]
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)


def get_config(folder_path, extra_cond=None):
    """Returns path to the config file in the folder."""

    if not os.path.exists(folder_path):
        raise Exception(f"Folder not found! <folder_path={folder_path}>")

    for root, _, files in os.walk(folder_path):
        if extra_cond is not None and extra_cond not in root:
            continue
        if "config.yaml" in files and ".hydra" not in root:
            return os.path.join(root, "config.yaml")


def get_latest_checkpoint_and_config(folder_path, extra_cond=None):
    """Returns latest checkpoint and config from a folder."""
    checkpoint_path = None
    config_path = None

    if not os.path.exists(folder_path):
        log.warning(f"Folder path not found! <folder_path={folder_path}>")
        return checkpoint_path, config_path

    # get latest checkpoint
    checkpoint_path = get_latest_checkpoint(folder_path, extra_cond=extra_cond)

    if not checkpoint_path:
        log.error("Checkpoint not found!")
        raise Exception(f"Checkpoint not found! <folder_path={folder_path}>")
        return checkpoint_path, config_path

    # get config from checkpoint
    config_path = get_config(folder_path, extra_cond=extra_cond)

    if not config_path:
        log.error("Config not found!")
        # raise Exception(f"Config not found! <folder_path={folder_path}>")
        return checkpoint_path, config_path

    return checkpoint_path, config_path


def configure_cfg_from_checkpoint(cfg):
    ckpt_path = Path(cfg.ckpt_path)
    if ckpt_path.is_dir():
        cfg.ckpt_path, cfg.config_path = get_latest_checkpoint_and_config(ckpt_path, extra_cond=cfg.filter_run)
        print(f"Using latest checkpoint: {cfg.ckpt_path}")
        print(f"Using config: {cfg.config_path}")

    if cfg.get("config_path"):
        with open(cfg.get("config_path"), "r") as stream:
            config = yaml.safe_load(stream)
        unflatten_config = unflatten_wandb_config(config)
        # update net config with config from checkpoint
        print("Updating model config.................")
        config = unflatten_config["model"]["net"]
        for k in cfg.model.net:
            if k in config and k not in ["_target_"]:
                print(f"Overwriting {k}: {cfg.model.net[k]} with {config[k]} (type: {type(config[k])})")
                if config[k] == "None":
                    config[k] = None
                cfg.model.net[k] = config[k]
        # update datamodule config with config from checkpoint
        print("Updating datamodule config.............")
        config = unflatten_config["datamodule"]
        for k in cfg.datamodule:
            if k in config and k in [
                "resize_size",
                "identity_frame",
                "channels",
                "use_latent",
                "latent_scale",
                "latent_type",
                "audio_rate",
                "separate_condition",
                # "exclude_dataset",
            ]:
                print(f"Overwriting {k}: {cfg.datamodule[k]} with {config[k]}")
                if config[k] == "None":
                    config[k] = None
                cfg.datamodule[k] = config[k]

    return cfg


def save_summary(model, save_path, logger):
    # Save model summary
    summary = model_summary.ModelSummary(model, -1)
    # Write the summary into a file
    with open(save_path + "/model_summary.txt", "w") as f:
        f.write(summary.__str__())
    # If logger is wandb, log the summary
    if logger is not None and len(logger):
        try:
            logger[0].experiment.save(save_path + "/model_summary.txt", policy="now")
        except TypeError:
            pass


def log_videos(img_cond, img_pred, logger, audio=None, prefix="", fps=25, sample_rate=16000):
    repeat_cond = repeat(img_cond, "c h w -> t c h w", t=img_pred.shape[0])
    grid = torch.cat([repeat_cond, img_pred], dim=-1).cpu() * 255.0
    temp_video_path = "temp.mp4"
    success = save_audio_video(
        grid,
        audio=audio,
        frame_rate=fps,
        sample_rate=sample_rate,
        save_path=temp_video_path,
        keep_intermediate=False,
    )
    logger.experiment.log(
        {
            f"{prefix}/generated_videos": wandb.Video(
                temp_video_path if success else grid,
                caption="diffused videos (condition left, generated right)",
                fps=fps,
            )
        },
    )
    if success:
        os.remove(temp_video_path)


def save_audio_video(
    video, audio=None, frame_rate=25, sample_rate=16000, save_path="temp.mp4", keep_intermediate=False
):
    """Save audio and video to a single file.
    video: (t, c, h, w)
    audio: (channels t)
    """
    save_path = str(save_path)
    try:
        torchvision.io.write_video("temp_video.mp4", rearrange(video, "t c h w -> t h w c"), frame_rate)
        video_clip = mpy.VideoFileClip("temp_video.mp4")
        if audio is not None:
            torchaudio.save("temp_audio.wav", audio, sample_rate)
            audio_clip = mpy.AudioFileClip("temp_audio.wav")
            video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(save_path, fps=frame_rate, codec="libx264", audio_codec="aac")
        if not keep_intermediate:
            os.remove("temp_video.mp4")
            if audio is not None:
                os.remove("temp_audio.wav")
        return 1
    except Exception as e:
        print(e)
        print("Saving video to file failed")
        return 0


def ask_until_response(question, expected=[]):
    while True:
        answer = input(question)
        if answer in expected:
            return answer
        else:
            print(f"Answer must be in {expected}!")


# do same einops operations on a list of tensors


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


def divisible_by(numer, denom):
    return (numer % denom) == 0
