import os
import numpy as np
from functools import partial
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import math
import decord
import time
from einops import repeat, rearrange
from more_itertools import sliding_window
from tqdm import tqdm
from src.utils.torch_utils import pad_n_stack_sequences
import torchaudio
import soundfile as sf
from src.utils.utils import ask_until_response
from src.utils.torch_utils import trim_pad_audio
from torchvision.transforms import RandomHorizontalFlip, ToTensor
from audiomentations import Compose, AddGaussianNoise, PitchShift
from PIL import Image

torchaudio.set_audio_backend("sox_io")
decord.bridge.set_bridge("torch")


# Similar to regular video dataset but trades flexibility for speed
class VideoDataset(Dataset):
    def __init__(
        self,
        filelist,
        resize_size=None,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        video_extension=".avi",
        audio_extension=".wav",
        audio_rate=16000,
        num_frames=5,
        need_cond=True,
        skip_short_videos_thresh=None,
        load_in_memory=False,
        step=1,
        allow_incomplete=False,
        short_video_format="replicate",
        mask_repeated_frames=False,
        identity_frame="first",
        separate_condition=False,
        max_missing_audio_files=10,
        scale_audio=False,
        split_audio_to_frames=True,
        augment=False,
        augment_audio=False,
        use_latent=False,
        latent_type="stable",
        latent_scale=1,
        exclude_dataset=[],
        use_serious_face=False,
        from_audio_embedding=False,
    ):

        self.use_serious_face = use_serious_face
        self.audio_folder = audio_folder
        self.from_audio_embedding = from_audio_embedding

        self.filelist = []
        self.audio_filelist = []
        missing_audio = 0
        with open(filelist, "r") as files:
            for f in files.readlines():
                f = f.rstrip()
                dataset_name = f.split("/")[-3]
                if dataset_name in exclude_dataset:
                    continue
                audio_path = f.replace(video_folder, audio_folder).replace(video_extension, audio_extension)
                if not os.path.exists(audio_path):
                    missing_audio += 1
                    print("Missing audio file: ", audio_path)
                    if missing_audio > max_missing_audio_files:
                        raise FileNotFoundError(f"Missing more than {max_missing_audio_files} audio files")
                    continue
                self.filelist += [f]
                self.audio_filelist += [audio_path]

        self.skip_short_videos = True if skip_short_videos_thresh is None else False

        if allow_incomplete and self.skip_short_videos:
            answer = ask_until_response(
                "Allow incomplete and skip short videos are both true. Are you sure you want to do this? (y/n)\n",
                ["y", "n"],
            )
            if answer == "y":
                print("Continuing")
            else:
                raise ValueError("Change config file and try again")

        self.resize_size = resize_size
        self.scale_audio = scale_audio
        self.step = step
        self.split_audio_to_frames = split_audio_to_frames
        self.separate_condition = separate_condition
        self.mask_repeated_frames = mask_repeated_frames
        self.allow_incomplete = allow_incomplete
        self.short_video_format = short_video_format

        self.augment = augment
        self.maybe_augment = RandomHorizontalFlip(p=0.5) if augment else lambda x: x
        self.maybe_augment_audio = (
            Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.002, p=0.25),
                    # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                    PitchShift(min_semitones=-1, max_semitones=1, p=0.25),
                    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.333),
                ]
            )
            if augment_audio
            else lambda x, sample_rate: x
        )
        self.maybe_augment_audio = partial(self.maybe_augment_audio, sample_rate=audio_rate)

        self.skip_short_videos_thresh = skip_short_videos_thresh
        self.need_cond = need_cond  # If need cond will extract one more frame than the number of frames
        # It is used for the conditional model when the condition is not on the temporal dimension
        num_frames = num_frames if not self.need_cond else num_frames + 1

        # Get metadata about video and audio
        # _, self.audio_rate = torchaudio.load(self.audio_filelist[0], channels_first=False)
        vr = decord.VideoReader(self.filelist[0])
        self.video_rate = math.ceil(vr.get_avg_fps())
        print(f"Video rate: {self.video_rate}")
        self.audio_rate = audio_rate
        a2v_ratio = self.video_rate / float(self.audio_rate)
        self.samples_per_frame = math.ceil(1 / a2v_ratio)

        self.num_frames = num_frames
        self.load_in_memory = load_in_memory
        self._indexes = self._get_indexes(self.filelist, self.audio_filelist)

        if load_in_memory:
            start_time = time.time()
            self.full_videos, self.full_audios = self.load_videos_and_audio_in_memory()
            print(f"Loaded {len(self.full_videos)} videos in {time.time() - start_time} seconds")

    def __len__(self):
        return len(self._indexes)

    def _load_audio(self, filename, max_len_sec, start=None):
        audio, sr = sf.read(
            filename,
            start=math.ceil(start * self.audio_rate),
            frames=math.ceil(self.audio_rate * max_len_sec),
            always_2d=True,
        )  # e.g (16000, 1)
        audio = audio.T  # (1, 16000)
        assert sr == self.audio_rate, f"Audio rate is {sr} but should be {self.audio_rate}"
        audio = audio.mean(0, keepdims=True)
        audio = self.maybe_augment_audio(audio)
        audio = torch.from_numpy(audio).float()
        # audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.audio_rate)
        audio = trim_pad_audio(audio, self.audio_rate, max_len_sec=max_len_sec)
        return audio[0]

    def _get_frames_and_audio(self, idx):
        indexes, video_file, audio_file = self._indexes[idx]
        audio_frames = None
        if self.load_in_memory:
            frames = self.full_videos[video_file][:, indexes]
            audio = self.full_audios[audio_file]
        else:
            vr = decord.VideoReader(video_file)
            frames = vr.get_batch(indexes).permute(3, 0, 1, 2).float()
            if not self.from_audio_embedding:
                audio = self._load_audio(
                    audio_file, max_len_sec=frames.shape[1] / self.video_rate, start=indexes[0] / self.video_rate
                )
            else:
                audio = torch.load(audio_file.split(".")[0] + "_emb.pt")
                audio_frames = audio[indexes, :]

        if audio_frames is None:
            audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)

        # audio_frames = audio_frames[indexes, :]

        n_frames = frames.shape[1]
        needed_frames = self.num_frames
        mask = torch.zeros(self.num_frames, dtype=bool)
        if n_frames < self.num_frames and not self.allow_incomplete:
            missing_frames = needed_frames - n_frames
            if self.mask_repeated_frames:
                mask[-missing_frames:] = True  # Mask out the added frames for temporal attention
            if self.short_video_format == "replicate":
                frames = torch.nn.ReplicationPad3d(
                    (0, 0, 0, 0, math.ceil(missing_frames / 2), math.floor(missing_frames / 2))
                )(frames)
                audio_frames = torch.nn.ReplicationPad1d(
                    (math.ceil(missing_frames / 2), math.floor(missing_frames / 2))
                )(audio_frames.T).T
            elif self.short_video_format == "repeat_last":
                frames = torch.cat([frames, repeat(frames[:, -1], "c h w -> c t h w", t=missing_frames)], dim=1)
                audio_frames = torch.cat([audio_frames, repeat(audio_frames[-1], "c -> t c", t=missing_frames)], dim=0)
            elif self.short_video_format == "repeat_first":
                frames = torch.cat([repeat(frames[:, 0], "c h w -> c t h w", t=missing_frames), frames], dim=1)
                audio_frames = torch.cat([repeat(audio_frames[0], "c -> c t", t=missing_frames), audio_frames], dim=0)
            else:
                raise NotImplementedError(
                    "short_video_format must be 'blacked', 'repeat_last', 'repeat_first' or 'from_image'"
                )
        # audio_frames = audio_frames.T
        audio_frames = audio_frames[1:]  # Remove audio of first frame
        assert (
            audio_frames.shape[0] == frames.shape[1] - 1
        ), f"{audio_frames.shape[0]} != {frames.shape[1]}, audio shape {audio_frames.shape}"
        if self.scale_audio:
            audio_frames = (audio_frames / audio_frames.max()) * 2 - 1

        if self.separate_condition:
            target = frames[:, 1:] if self.need_cond else frames
            identity = target[:, 0] / 255.0
            target = self.scale_and_crop((target / 255.0) * 2 - 1)
        else:
            frames = self.scale_and_crop((frames / 255.0) * 2 - 1)
            target = frames[:, 1:] if self.need_cond else frames
            identity = target[:, 0]

        if self.use_serious_face:
            # Useful to test if model works with serious face as input
            # Replace the identity frame by a a frame from the same speaker but serious
            identity = self.get_serious_face(audio_file)

        if not self.split_audio_to_frames:
            audio_frames = rearrange(audio_frames, "f s -> (f s)")

        return identity, target, mask, audio_frames

    def get_serious_face(self, original_file):
        speaker = os.path.basename(original_file).split("_")[0].split("-")[0]
        dir_path = os.path.dirname(original_file).replace(self.audio_folder, "Serious_Face")
        file = os.path.join(dir_path, f"{speaker}.png")
        try:
            img = ToTensor()(Image.open(file))
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file}, only supports mahnob for now")
        return self.scale_and_crop(img.unsqueeze(1) * 2 - 1).squeeze(1)

    def load_videos_and_audio_in_memory(self):
        videos = {}
        audios = {}
        for idx in tqdm(range(len(self.filelist)), desc="Loading videos in memory"):
            file = self.filelist[idx]
            audio_file = self.audio_filelist[idx]
            vr = decord.VideoReader(file)
            frames = vr.get_batch(range(len(vr))).permute(3, 0, 1, 2).float()
            audio = self._load_audio(audio_file, max_len_sec=len(vr) / self.video_rate)
            videos[file] = frames
            audios[audio_file] = audio
        return videos, audios

    def _get_indexes(self, video_filelist, audio_filelist):
        indexes = []
        for vid_file, audio_file in zip(video_filelist, audio_filelist):
            vr = decord.VideoReader(vid_file)
            len_video = len(vr)
            # Short videos
            if len_video < self.num_frames:
                if not self.skip_short_videos:
                    if len_video >= self.skip_short_videos_thresh:
                        indexes.append((range(len_video), vid_file, audio_file))
                else:
                    continue
            else:
                possible_indexes = list(sliding_window(range(len_video), self.num_frames))[:: self.step]
                possible_indexes = list(map(lambda x: (x, vid_file, audio_file), possible_indexes))
                indexes.extend(possible_indexes)
        print("Indexes", len(indexes), "\n")
        return indexes

    def scale_and_crop(self, video):
        h, w = video.shape[-2], video.shape[-1]
        # scale shorter side to resolution

        if self.resize_size is not None:
            scale = self.resize_size / min(h, w)
            if h < w:
                target_size = (self.resize_size, math.ceil(w * scale))
            else:
                target_size = (math.ceil(h * scale), self.resize_size)
            video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False, antialias=True)

            # center crop
            h, w = video.shape[-2], video.shape[-1]
            w_start = (w - self.resize_size) // 2
            h_start = (h - self.resize_size) // 2
            video = video[:, :, h_start : h_start + self.resize_size, w_start : w_start + self.resize_size]
        return self.maybe_augment(video)

    def __getitem__(self, idx):

        identity, target, mask, audio = self._get_frames_and_audio(idx)
        video_file = self._indexes[idx][1]
        # print("Target", target.shape)
        # print("Mask", mask.shape)
        # print("Audio", audio.shape)
        # print("Identity", identity.shape)

        return {"identity": identity, "video": target, "mask": mask, "audio": audio, "video_file": video_file}


def collate(batch):
    identity_list = []
    target_list = []
    mask_list = []
    audio_list = []

    for item in batch:
        identity_list.append(item["identity"])
        target_list.append(rearrange(item["video"], "c t h w -> t c h w"))
        mask_list.append(item["mask"])
        audio_list.append(item["audio"])

    target, lengths, order = pad_n_stack_sequences(target_list)
    mask, _ = pad_n_stack_sequences(mask_list, order=order)

    identity_list = torch.stack([identity_list[i] for i in order])
    audio, audio_lengths = pad_n_stack_sequences(audio_list, order=order)
    # mask_list = torch.stack([mask_list[i] for i in order])

    batch = {
        "identity": identity_list,
        "video": rearrange(target, "b t c h w -> b c t h w"),
        "lengths": lengths,
        "mask": mask,
        "audio": audio,
        "audio_lengths": audio_lengths,
    }

    return batch


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import cv2

    transform = transforms.Compose(transforms=[transforms.Resize((256, 256))])
    dataset = VideoDataset(
        "/vol/paramonos2/projects/antoni/datasets/mahnob/filelist_videos_val.txt", transform=transform, num_frames=25
    )
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))

    for i in range(10):
        print(dataset[i][0].shape, dataset[i][1].shape)

    image_identity = (dataset[idx][0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    image_other = (dataset[idx][1][:, -1].permute(1, 2, 0).numpy() + 1) / 2 * 255
    cv2.imwrite("image_identity.png", image_identity[:, :, ::-1])
    for i in range(25):
        image = (dataset[idx][1][:, i].permute(1, 2, 0).numpy() + 1) / 2 * 255
        cv2.imwrite(f"tmp_vid_dataset/image_{i}.png", image[:, :, ::-1])
