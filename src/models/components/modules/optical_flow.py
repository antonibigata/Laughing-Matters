from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
import torch
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


class OpticalFlow:
    def __init__(self, model_type="pytorch", size=[520, 960]):
        self.model_type = model_type
        self.size = size
        assert model_type in ["pytorch", "opencv"]
        if model_type == "pytorch":
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).cuda()
            self.model.eval()
            self.transforms = Raft_Large_Weights.DEFAULT.transforms()
            for param in self.model.parameters():
                param.requires_grad = False

        self.magnitude = defaultdict(list)

    def preprocess(self, img1_batch, img2_batch, size=[520, 960]):
        """Preprocesses the images for optical flow estimation.
        Args:
            img1_batch (torch.Tensor): Batch of images of shape (B, C, H, W). (0, 255)
            img2_batch (torch.Tensor): Batch of images of shape (B, C, H, W). (0, 255)
            size (list): Size of the images after resizing.
        """
        img1_batch = F.resize(img1_batch, size=size, antialias=False)
        img2_batch = F.resize(img2_batch, size=size, antialias=False)
        if self.model_type == "pytorch":
            return self.transforms(img1_batch, img2_batch)
        elif self.model_type == "opencv":
            return img1_batch, img2_batch
        else:
            raise ValueError("Model type not supported.")

    def cuda(self):
        return self

    def get_optical_flow_magnitude(self, img1_batch, img2_batch):
        """Estimates the optical flow between two images.
        Args:
            video (torch.Tensor): Batch of images of shape (T, C, H, W). (0, 255)
        """
        if self.model_type == "pytorch":
            with torch.no_grad():
                flow = self.model(img1_batch.cuda(), img2_batch.cuda())[-1]
                magnitude, _ = cv2.cartToPolar(flow[:, 0].cpu().numpy(), flow[:, 1].cpu().numpy())
        elif self.model_type == "opencv":
            magnitude = []
            for img1, img2 in zip(img1_batch, img2_batch):  # iterate over batch
                # convert to grayscale
                img1 = cv2.cvtColor(img1.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(img2.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
                # estimate optical flow
                flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magn, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude.append(magn)
            magnitude = np.array(magnitude)

        return magnitude

    def update(self, videos, video_paths=None):
        """Updates the optical flow model.
        Args:
            video (torch.Tensor): Batch of images of shape (T, C, H, W). (0, 255)
        """
        if video_paths is None:
            video_paths = [None] * len(videos)
        for i in range(len(video_paths)):
            video_path = video_paths[i]
            video = videos[i]
            if video_path is not None:
                speaker = Path(video_path).stem
                speaker = speaker.split("_")[0].split("-")[0]
            else:
                speaker = "unknown"
            img1 = video[:-1]
            img2 = video[1:]
            img1_batch, img2_batch = self.preprocess(img1, img2, size=self.size)
            magnitude = self.get_optical_flow_magnitude(img1_batch, img2_batch)
            self.magnitude[speaker] += [np.mean(magnitude, axis=0) / np.linalg.norm(magnitude)]

    def compute(self):
        """Computes the optical flow."""
        for speaker in self.magnitude:
            self.magnitude[speaker] = np.mean(self.magnitude[speaker], axis=0)

    def save_figures(self, path_dir):
        """Saves the optical flow figures.
        Args:
            path (str): Path to save the optical flow figures.
        """
        Path(path_dir).mkdir(parents=True, exist_ok=True)
        all_speakers = []
        for speaker in self.magnitude:
            plt.figure()
            plt.imshow(self.magnitude[speaker])
            plt.axis("off")
            all_speakers.append(self.magnitude[speaker])
            plt.savefig(f"{path_dir}/{speaker}_{self.model_type}.png", bbox_inches="tight")
            plt.close()
        plt.figure()
        all_speakers = np.mean(all_speakers, axis=0)
        plt.imshow(all_speakers)
        plt.axis("off")
        plt.savefig(f"{path_dir}/all_{self.model_type}.png", bbox_inches="tight")
        plt.close()
