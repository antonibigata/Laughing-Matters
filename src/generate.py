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

import hydra
from omegaconf import DictConfig
from torchvision.transforms import transforms
import torchaudio
import torch
from einops import rearrange
import cv2
from pathlib import Path
from tqdm import tqdm
import math

from src.utils.torch_utils import load_checkpoint, trim_pad_audio
from src import utils
from src.utils.utils import configure_cfg_from_checkpoint, log_videos, default, save_summary
from src.utils.face_aligment.alignment_module import FaceAligment
from src.utils.diffusion_utils import unnormalize_to_zero_to_one

log = utils.get_pylogger(__name__)
FPS = 25


def generate(cfg: DictConfig):

    assert cfg.ckpt_path

    cfg = configure_cfg_from_checkpoint(cfg)

    # Load model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    net = hydra.utils.instantiate(cfg.model.net)

    net = load_checkpoint(
        net,
        cfg.ckpt_path,
        allow_extra_keys=True,
        extra_key="state_dict",
        replace=("model.", ""),
        map_location="cuda",
    )
    cfg.model.num_frames_val = default(cfg.n_generate_frames, cfg.model.num_frames_val)
    model = hydra.utils.instantiate(cfg.model, sample_rate=cfg.datamodule.get("audio_rate"), video_rate=cfg.fps).to(
        "cuda"
    )
    # model.load_from_checkpoint(cfg.ckpt_path, net=net)
    model.update_inf_model(net.to("cuda"))

    log.info("Instantiating loggers...")
    logger = utils.instantiate_loggers(cfg.get("logger"))
    save_summary(model, cfg.paths.output_dir, logger)
    if isinstance(logger, list) and len(logger) > 0:
        logger = logger[0]

    # Load face processor
    face_processor = FaceAligment(
        cfg.mean_face_path,
        scale=1.0,
        device="cuda",
        fps=cfg.fps,
        offset=cfg.offset,
        out_size=None,
        square_it=True,
        reference=None,
    )

    # Process image
    image_path = Path(cfg.image_path)
    image = face_processor.process(cfg.image_path)
    tensor_resize = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((cfg.datamodule.resize_size, cfg.datamodule.resize_size))]
    )
    image = tensor_resize(image).to("cuda") * 2 - 1
    if cfg.debug:
        # Save image
        saved_image = (unnormalize_to_zero_to_one(image.cpu().numpy()) * 255).astype("uint8")
        saved_image = rearrange(saved_image, "c h w -> h w c")
        cv2.imwrite(str(image_path.parent / f"{image_path.stem}_processed.png"), saved_image[:, :, ::-1])

    audio = None
    if cfg.audio_path is not None:
        audio, sr = torchaudio.load(cfg.audio_path, channels_first=True)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=cfg.datamodule.get("audio_rate"))[0]
        samples_per_frame = math.ceil(cfg.datamodule.get("audio_rate") / FPS)
        n_frames = audio.shape[-1] / samples_per_frame
        if not n_frames.is_integer():
            print("Audio shape before trim_pad_audio: ", audio.shape)
            audio = trim_pad_audio(
                audio, cfg.datamodule.get("audio_rate"), max_len_raw=math.ceil(n_frames) * samples_per_frame
            )
            print("Audio shape after trim_pad_audio: ", audio.shape)
        audio = rearrange(audio, "(f s) -> f s", s=samples_per_frame)

    pred = model(
        cond=image.unsqueeze(0),
        cond_scale=cfg.model.cond_scale,
        autoregressive_passes=cfg.autoregressive_passes,
        audio=audio.unsqueeze(0).cuda(),
    )
    assert not torch.isnan(pred).any(), "NaNs in prediction"

    # Save video
    if cfg.save_video and (logger is None or (isinstance(logger, list) and len(logger) > 0)):
        pred = rearrange(pred[0], "c t h w -> t h w c").cpu().numpy()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            str((Path(cfg.log_folder) / image_path.name).with_suffix(".mp4")),
            fourcc,
            cfg.fps,
            (pred.shape[2], pred.shape[1]),
        )
        for frame in tqdm(pred, desc="Saving video"):
            video.write((frame * 255).astype("uint8")[..., ::-1])
        video.release()
    else:
        # Logging video to wandb
        pred = rearrange(pred[0], "c t h w -> t c h w")
        audio_cut = None
        if audio is not None:
            audio_cut = audio[: pred.shape[0]]  # Cut audio to match video length
            audio_cut = rearrange(audio_cut, "... -> (...)").unsqueeze(0)
        log_videos(
            unnormalize_to_zero_to_one(image),
            pred,
            logger,
            audio=audio_cut,
            prefix="gen",
            fps=FPS,
            sample_rate=cfg.datamodule.get("audio_rate"),
        )


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="generate.yaml")
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
