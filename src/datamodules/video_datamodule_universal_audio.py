from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)
from src.datamodules.components.video_dataset_universal_audio import VideoDataset, collate


class VideoDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        file_list_train: str,
        file_list_val: str = None,
        file_list_test: str = None,
        resize_size: int = 256,
        identity_frame: str = "first",
        short_video_format: str = "repeat_first",
        load_in_memory: bool = False,
        num_frames: int = 16,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        skip_short_videos_thresh=None,
        mask_repeated_frames=False,
        channels=3,
        condition_dim: str = "channels",
        allow_incomplete: bool = False,
        separate_condition: bool = False,
        step: int = 1,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        video_extension=".avi",
        audio_extension=".wav",
        audio_rate=16000,
        max_missing_audio_files=10,
        scale_audio=False,
        split_audio_to_frames=True,
        augment: bool = False,
        augment_audio: bool = False,
        exclude_dataset: list = [],
        use_serious_face: bool = False,
        from_audio_embedding: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.need_cond = condition_dim != "time"

        self.collate_fn = collate if allow_incomplete else None

        # self.transform = transforms.Compose([transforms.Resize((resize_size, resize_size), antialias=True)])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = VideoDataset(
                self.hparams.file_list_train,
                resize_size=self.hparams.resize_size,
                identity_frame=self.hparams.identity_frame,
                num_frames=self.hparams.num_frames,
                short_video_format=self.hparams.short_video_format,
                load_in_memory=self.hparams.load_in_memory,
                skip_short_videos_thresh=self.hparams.skip_short_videos_thresh,
                mask_repeated_frames=self.hparams.mask_repeated_frames,
                need_cond=self.need_cond,
                allow_incomplete=self.hparams.allow_incomplete,
                step=self.hparams.step,
                separate_condition=self.hparams.separate_condition,
                audio_folder=self.hparams.audio_folder,
                video_folder=self.hparams.video_folder,
                video_extension=self.hparams.video_extension,
                audio_extension=self.hparams.audio_extension,
                audio_rate=self.hparams.audio_rate,
                scale_audio=self.hparams.scale_audio,
                max_missing_audio_files=self.hparams.max_missing_audio_files,
                split_audio_to_frames=self.hparams.split_audio_to_frames,
                augment=self.hparams.augment,
                augment_audio=self.hparams.augment_audio,
                exclude_dataset=self.hparams.exclude_dataset,
                from_audio_embedding=self.hparams.from_audio_embedding,
            )
            if self.hparams.file_list_val:
                self.data_val = VideoDataset(
                    self.hparams.file_list_val,
                    resize_size=self.hparams.resize_size,
                    identity_frame=self.hparams.identity_frame,
                    num_frames=self.hparams.num_frames,
                    short_video_format=self.hparams.short_video_format,
                    load_in_memory=self.hparams.load_in_memory,
                    skip_short_videos_thresh=self.hparams.skip_short_videos_thresh,
                    mask_repeated_frames=self.hparams.mask_repeated_frames,
                    need_cond=self.need_cond,
                    allow_incomplete=self.hparams.allow_incomplete,
                    step=self.hparams.step,
                    separate_condition=self.hparams.separate_condition,
                    audio_folder=self.hparams.audio_folder,
                    video_folder=self.hparams.video_folder,
                    video_extension=self.hparams.video_extension,
                    audio_extension=self.hparams.audio_extension,
                    audio_rate=self.hparams.audio_rate,
                    scale_audio=self.hparams.scale_audio,
                    max_missing_audio_files=self.hparams.max_missing_audio_files,
                    split_audio_to_frames=self.hparams.split_audio_to_frames,
                    augment=False,
                    augment_audio=False,
                    exclude_dataset=self.hparams.exclude_dataset,
                    use_serious_face=self.hparams.use_serious_face,
                    from_audio_embedding=self.hparams.from_audio_embedding,
                )
            if self.hparams.file_list_test:
                self.data_test = VideoDataset(
                    self.hparams.file_list_test,
                    resize_size=self.hparams.resize_size,
                    identity_frame=self.hparams.identity_frame,
                    num_frames=self.hparams.num_frames,
                    short_video_format=self.hparams.short_video_format,
                    load_in_memory=self.hparams.load_in_memory,
                    skip_short_videos_thresh=self.hparams.skip_short_videos_thresh,
                    mask_repeated_frames=self.hparams.mask_repeated_frames,
                    need_cond=self.need_cond,
                    allow_incomplete=self.hparams.allow_incomplete,
                    step=self.hparams.step,
                    separate_condition=self.hparams.separate_condition,
                    audio_folder=self.hparams.audio_folder,
                    video_folder=self.hparams.video_folder,
                    video_extension=self.hparams.video_extension,
                    audio_extension=self.hparams.audio_extension,
                    audio_rate=self.hparams.audio_rate,
                    scale_audio=self.hparams.scale_audio,
                    max_missing_audio_files=self.hparams.max_missing_audio_files,
                    split_audio_to_frames=self.hparams.split_audio_to_frames,
                    augment=False,
                    augment_audio=False,
                    exclude_dataset=self.hparams.exclude_dataset,
                    use_serious_face=self.hparams.use_serious_face,
                    from_audio_embedding=self.hparams.from_audio_embedding,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        else:
            return None

    def test_dataloader(self):
        if self.data_test:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        else:
            return None

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "video_datamodule_universal_audio.yaml")
    # cfg.data_dir = str(root / "data")
    data = hydra.utils.instantiate(cfg)
    data.prepare_data()
    data.setup()
