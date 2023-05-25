import os
from typing import Any
from enum import Enum
from einops import rearrange, repeat
import wandb
import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only


from src.models.components.modules.augment import AugmentPipe
from src.utils.diffusion_utils import unnormalize_to_zero_to_one
from src.utils.utils import default
from src.models.components.modules.metrics import FVD
from PIL import Image
from pathlib import Path

from src import utils
from src.utils.utils import save_audio_video

log = utils.get_pylogger(__name__)


class Status(Enum):
    SANITY = 0
    FIRST_LOOP = 1
    MAIN_LOOP = 2


def identity(t, *args, **kwargs):
    return t


class DiffusionModule(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        image_size,
        loss_fn=F.l1_loss,
        calculate_fid=False,
        n_images_fid=100,
        debug=False,
        sample_and_save_every=1000,
        ema_module=None,
        wandb_fps=None,
        batch_size=None,
        record_grads=False,
        # Sampling
        autoregressive_passes=1,
        num_frames_val=None,
        # Related to michal code
        vlb_weight=0.0,
        lip_weight=0.0,
        # Related to video
        prob_focus_present=0.0,
        # Related to augment pipe
        augment_proba=0,
        # Conditioning
        cond_scale=1,
        condition_dim="channels",
        unconditional_percent=0.0,
        null_cond_prob=0.0,
        # FVD
        calculate_fvd=False,
        fvd_every=100,
        # Conditional noise
        add_conditional_noise=None,
        # metadata
        sample_rate=16000,
        video_rate=25,
        # VAE
        use_latent=False,
        latent_scale=1,
        latent_type="stable",
        max_chunk_decode=10,  # 10 represent an increase og 5GB of memory
        # Misc
        n_devices=1,
        eval_save_all=False,  # If True, will save all video generated during evaluation
        compile_model=True,
        n_frames_loss=None,  # Number of frames to use for loss calculation
    ):
        super().__init__()

        torch.set_float32_matmul_precision("medium")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.train_status = Status.SANITY
        self.log_media = False
        self.wandb_fps = default(wandb_fps, 4)
        self.sample_rate = sample_rate
        self.video_rate = video_rate
        self.do_sync_dist = True if n_devices > 1 else False

        augment_pipe = None
        if augment_proba > 0:
            augment_kwargs = dict(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
            augment_pipe = AugmentPipe(augment_proba, **augment_kwargs)

        # if compile_model:
        #     print("Compiling model")
        #     net = torch.compile(net, mode="max-autotune")

        self.model = net

        if ema_module is not None:
            self.ema_model = ema_module(self.model)
        else:
            self.ema_model = None

        self.diffusion_model = diffusion_model(
            net,
            loss_fn,
            image_size,
            inference_model=self.ema_model,
            augment_pipe=augment_pipe,
            unconditional_percent=unconditional_percent if condition_dim == "time" else None,
            add_conditional_noise=add_conditional_noise,
            use_latent=use_latent,
            null_cond_prob=null_cond_prob,
        )
        self.diffusion_model.num_frames = default(
            num_frames_val, self.diffusion_model.num_frames
        )  # Potentially inference with more frames

        self.image_size = image_size

        self.loss_fn = loss_fn

        self.prev_global_step = 0

        # # metric objects for calculating and averaging accuracy across batches
        self.calculate_fid = calculate_fid
        if calculate_fid:
            self.test_fid = FrechetInceptionDistance(64, reset_real_features=False)

        self.calculate_fvd = calculate_fvd
        self.fvd_every = fvd_every if calculate_fvd else None
        if self.fvd_every is not None:
            self.val_fvd = FVD(reset_real_features=False, resize=224)
            self.test_fvd = FVD(reset_real_features=False, resize=224)

        # # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.calculate_real = True  # For FID, FVD

        # Misc
        self.record_grads = record_grads
        self.eval_save_all = eval_save_all

        # VAE
        self.use_latent = use_latent
        self.max_chunk_decode = max_chunk_decode

    def update_inf_model(self, model):
        self.diffusion_model.update_inf_model(model)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        preds = self.diffusion_model.sample(*args, **kwargs)

        if self.use_latent:
            preds = unnormalize_to_zero_to_one(self.decode_video(preds)).clamp(0, 1)
        return preds

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        # self.val_acc_best.reset()
        if self.logger is not None and self.record_grads:
            log.info("Recording gradients")
            self.logger.watch(self.model, log_freq=500)
        if self.calculate_fid:
            self.val_fid.reset()
        self.train_status = Status.FIRST_LOOP

    def step(self, img_cond, img_target, mask, lengths, audio):
        (
            _,
            _,
            h,
            w,
            _,
            img_size,
        ) = (
            *img_cond.shape,
            img_cond.device,
            self.image_size,
        )
        assert h == img_size and w == img_size, f"height and width of image must be {img_size}"
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        return self.diffusion_model.diffusion_step(
            img_target,
            img_cond,
            audio=audio,
            lengths=lengths,
            prob_focus_present=self.hparams.prob_focus_present,
            focus_present_mask=mask,
            n_frames_loss=self.hparams.n_frames_loss,
        )

    def log_input_model(self, batch):
        log_dir = "logs/debug_input_model"
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        img_cond, img_target, _, _, _ = self.read_batch(batch)
        img_cond = (unnormalize_to_zero_to_one(img_cond).cpu() * 255).to(torch.uint8)
        img_target = (unnormalize_to_zero_to_one(img_target).cpu() * 255).to(torch.uint8)
        for i in range(img_cond.shape[0]):
            Image.fromarray(img_cond[i].permute(1, 2, 0).numpy().squeeze()).save(f"{log_dir}/img_cond_{i}.png")
            if len(img_target.shape) == 5:
                for j in range(img_target.shape[2]):
                    Image.fromarray(img_target[i, :, j].permute(1, 2, 0).numpy().squeeze()).save(
                        f"{log_dir}/img_target_{i}_{j}.png"
                    )
            else:
                Image.fromarray(img_target[i].permute(1, 2, 0).numpy().squeeze()).save(f"{log_dir}/img_target_{i}.png")

    def training_step(self, batch: Any, batch_idx: int):
        img_cond, img_target, mask, lengths, audio = self.read_batch(batch)

        if self.hparams.debug:
            self.log_input_model(batch)

        loss, loss_dict = self.step(img_cond, img_target, mask, lengths, audio)

        if (
            self.global_step
            and self.global_step % self.hparams.sample_and_save_every == 0
            and self.prev_global_step != self.global_step  # Only log media once if accumulate_grad_batches > 1
        ):
            self.log_media = True
            preds = self.sample(
                cond=img_cond,
                cond_scale=self.hparams.cond_scale,
                batch_size=16,
                audio=audio,
                # lengths=lengths,
                autoregressive_passes=self.hparams.autoregressive_passes,
                training_sample=True,
            )
            self.log_videos(batch, preds, "train")
            self.prev_global_step = self.global_step

        # update and log metrics
        self.train_loss(loss)
        if len(loss_dict) > 1:
            for k, v in loss_dict.items():
                self.log(
                    f"train/{k}", v, batch_size=self.hparams.batch_size, prog_bar=True, sync_dist=self.do_sync_dist
                )
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, batch_size=self.hparams.batch_size, prog_bar=True)

        return {"loss": loss, "loss_dict": loss_dict}

    def read_batch(self, batch):
        img_cond = batch["identity"]
        img_target = batch["video"]
        mask = batch.get("mask", None)
        lengths = batch.get("lengths", None)
        audio = batch.get("audio", None)

        return img_cond, img_target, mask, lengths, audio

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema_model is not None:
            self.ema_model.update()

    @rank_zero_only
    def log_videos(self, batch, pred, prefix="", log_all=False):
        n_frames = pred.shape[2]
        img_cond_batch, img_target_batch, _, _, audio_batch = self.read_batch(batch)
        video_files = batch.get("video_file", [""] * pred.shape[0])
        if log_all:
            random_idxs = [i for i in range(pred.shape[0])]
        elif prefix == "test":
            random_idxs = [0]
        else:
            random_idxs = [torch.randint(0, pred.shape[0], (1,)).item()]

        for random_idx in random_idxs:
            if len(audio_batch.shape) == 4:
                audio = None
                print("Warning: audio is embedding and cannot be logged")
            if audio_batch is not None:
                audio = audio_batch.cpu()[random_idx].unsqueeze(0)
                audio = rearrange(audio, "b ... -> b (...)")

            if self.use_latent:
                img_target = img_target_batch[random_idx]
                img_target = self.decode_video(rearrange(img_target, "c t h w -> () c t h w")).squeeze()
                img_target = unnormalize_to_zero_to_one(img_target).clamp(0, 1)
            else:
                img_target = unnormalize_to_zero_to_one(img_target_batch)[random_idx]
                # img_cond = unnormalize_to_zero_to_one(img_cond)[random_idx]
            if img_target.shape[1] != pred.shape[2]:  # Replace video with conditional image
                img_cond = unnormalize_to_zero_to_one(img_cond_batch)[random_idx]
                img_target = repeat(img_cond, "c h w -> c t h w", t=pred.shape[2])

            img_pred = rearrange(pred[random_idx], "c t h w -> t c h w")
            img_target = rearrange(img_target, "c t h w -> t c h w")
            # img_cond = repeat(img_cond, "c h w -> t c h w", t=img_pred.shape[0])
            if self.eval_save_all:
                video = img_pred.cpu() * 255.0  # Only prediction
            else:
                video = torch.cat([img_target, img_pred], dim=-1).cpu() * 255.0

            if video.shape[1] == 1:
                video = video.repeat(1, 3, 1, 1)

            temp_video_path = "temp.mp4"
            if self.eval_save_all:
                video_file = Path(video_files[random_idx]).stem
                temp_video_path = f"{self.logger.save_dir}/{video_file}_{prefix}_{random_idx}_video.mp4"
            success = save_audio_video(
                video,
                audio=audio,
                frame_rate=self.video_rate,
                sample_rate=self.sample_rate,
                save_path=temp_video_path,
                keep_intermediate=False,
            )

            if self.logger is not None and not self.eval_save_all:  # Don't save to wandb if saving all videos
                self.logger.experiment.log(
                    {
                        f"{prefix}/generated_videos": wandb.Video(
                            temp_video_path if success else video,
                            caption=f"diffused videos w {n_frames} frames (condition left, generated right)",
                            fps=self.wandb_fps,
                            format="mp4",
                        )
                    },
                )
            if success and not self.eval_save_all:  # Don't delete if saving all videos
                os.remove(temp_video_path)

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(
    # set_to_none=True
    #     )  # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html#set-grads-to-none

    def validation_step(self, batch: Any, batch_idx: int):
        img_cond, img_target, mask, lengths, audio = self.read_batch(batch)

        loss, loss_dict = self.step(img_cond, img_target, mask, lengths, audio)

        preds = None
        if batch_idx == 0 and self.log_media:
            self.log_media = False
            preds = self.sample(
                cond=img_cond,
                cond_scale=self.hparams.cond_scale,
                batch_size=16,
                audio=audio,
                # lengths=lengths,
                autoregressive_passes=self.hparams.autoregressive_passes,
                training_sample=True,
            )
            self.log_videos(batch, preds, "val")

        if self.current_epoch and self.fvd_every and self.current_epoch % self.fvd_every == 0:
            if preds is None:
                preds = self.sample(
                    cond=img_cond,
                    cond_scale=self.hparams.cond_scale,
                    batch_size=16,
                    audio=audio,
                    # lengths=lengths,
                    autoregressive_passes=1,
                    training_sample=True,
                )
            self.update_fvd(batch, preds, "val", update_real=self.calculate_real)

        self.val_loss(loss)
        if len(loss_dict) > 1:
            for k, v in loss_dict.items():
                self.log(f"val/{k}", v, batch_size=self.hparams.batch_size, sync_dist=self.do_sync_dist)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, batch_size=self.hparams.batch_size, prog_bar=True)

        return {"loss": loss, "loss_dict": loss_dict}

    def on_validation_epoch_end(self):
        if self.current_epoch and self.fvd_every and self.current_epoch % self.fvd_every == 0:
            self.calculate_real = False
            self.log("val/fvd", self.val_fvd, prog_bar=True, batch_size=self.hparams.batch_size)

    def test_step(self, batch: Any, batch_idx: int):
        img_cond, img_target, mask, lengths, audio = self.read_batch(batch)

        loss = 0
        loss_dict = {}
        if not self.eval_save_all:
            loss, loss_dict = self.step(img_cond, img_target, mask, lengths, audio)

        if batch_idx == 0 or self.eval_save_all:
            preds = self.sample(
                cond=img_cond,
                cond_scale=self.hparams.cond_scale,
                batch_size=16,
                audio=audio,
                # lengths=lengths,
                autoregressive_passes=self.hparams.autoregressive_passes,
            )
            if self.eval_save_all:
                self.log_videos(batch, preds, f"evaluation_{batch_idx}", log_all=True)
            else:
                self.log_videos(batch, preds, "test")

        # calculate FID
        if self.calculate_fid:
            preds = self.sample(
                cond=img_cond,
                cond_scale=self.hparams.cond_scale,
                batch_size=img_cond.shape[0],
                audio=audio,
                # lengths=lengths,
                autoregressive_passes=1,
            )
            self.update_fid(batch, preds, "test")

        if self.calculate_fvd:
            preds = self.sample(
                cond=img_cond,
                cond_scale=self.hparams.cond_scale,
                batch_size=img_cond.shape[0],
                audio=audio,
                # lengths=lengths,
                autoregressive_passes=1,
            )
            self.update_fvd(batch, preds, "test", update_real=True)

        # update and log metrics
        self.test_loss(loss)
        if len(loss_dict) > 1:
            for k, v in loss_dict.items():
                self.log(f"val/{k}", v, batch_size=self.hparams.batch_size, sync_dist=self.do_sync_dist)
        # self.val_acc(preds, targets)
        self.log("test/loss", self.test_loss, batch_size=self.hparams.batch_size, prog_bar=True)

        return {"loss": loss, "loss_dict": loss_dict}

    def update_fvd(self, batch, preds, prefix, update_real=True):
        _, img_target, _, _, _ = self.read_batch(batch)
        batch_imgs = unnormalize_to_zero_to_one(img_target)
        if update_real:
            getattr(self, f"{prefix}_fvd").update(batch_imgs, real=True)
        getattr(self, f"{prefix}_fvd").update(preds, real=False)

    def update_fid(self, batch, preds, prefix, update_real=True):
        _, img_target, _, _, _ = self.read_batch(batch)
        batch_imgs = unnormalize_to_zero_to_one(img_target)
        if len(batch_imgs.shape) > 4:
            preds = rearrange(preds, "b c t h w -> (b t) c h w")
            batch_imgs = rearrange(batch_imgs, "b c t h w -> (b t) c h w")
        if update_real:
            getattr(self, f"{prefix}_fid").update((batch_imgs * 255.0).to(torch.uint8), real=True)
        getattr(self, f"{prefix}_fid").update((preds * 255.0).to(torch.uint8), real=False)

    def on_test_epoch_end(self):
        if self.calculate_fid:
            self.log("test/fid", self.test_fid, prog_bar=True, batch_size=self.hparams.batch_size)
            self.test_fid.reset()  # Normally not needed, but just to be sure
        if self.calculate_fvd:
            self.log("test/fvd", self.test_fvd, prog_bar=True, batch_size=self.hparams.batch_size)
            self.test_fvd.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None and self.hparams.scheduler.func is not None:
            if self.hparams.scheduler.func == torch.optim.lr_scheduler.StepLR:
                if self.hparams.n_epochs is not None:
                    step_size = self.hparams.n_epochs // 18
                    print(f"Scheduler {step_size}")
                else:
                    step_size = 166
                scheduler = self.hparams.scheduler(optimizer=optimizer, step_size=step_size)
            elif self.hparams.scheduler.func == torch.optim.lr_scheduler.CosineAnnealingLR:
                scheduler = self.hparams.scheduler(optimizer=optimizer, T_max=self.trainer.estimated_stepping_batches)
            else:
                # iter_per_epoch = len(self.trainer.datamodule.train_dataloader()) / self.trainer.accumulate_grad_batches
                # scheduler = partial(self.hparams.scheduler, iter_per_epoch=iter_per_epoch)
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer, max_iters=self.trainer.estimated_stepping_batches
                )
            #          scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
            # )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "diffusion.yaml")
    _ = hydra.utils.instantiate(cfg)
