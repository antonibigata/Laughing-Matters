import os
import numpy as np
import seaborn as sns
from skimage import io
import face_alignment
from src.utils.face_aligment.landmarks_smoother import LandmarkSmootherRTS
from src.utils.math_utils import furthest_away_pts
from pathlib import Path


class LandmarksExtractor:
    def __init__(self, device="cuda", fps=25, apply_smoothing=True, landmarks_type="2D", flip=False):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D if landmarks_type == "2D" else face_alignment.LandmarksType._3D,
            flip_input=flip,
            device=device,
            face_detector="sfd",
        )
        self.apply_smoothing = apply_smoothing

        self.smoother = LandmarkSmootherRTS(fps=fps, process_noise=500, detector_accuracy=1, ignore_value=np.nan,)

        self.amplitude = []
        self.mov_std = []
        self.mar = []
        self.landmarks = []
        self.stable_pt = 33  # 33 is the nose tip

    def cuda(self):
        return self

    def extract_landmarks(self, image):
        # image: either a path to an image or a numpy array (H, W, C) or tensor batch  (B, C, H, W)
        if isinstance(image, str):
            image = io.imread(image)
        if len(image.shape) == 3:
            preds = self.fa.get_landmarks(image)
        else:
            preds = self.fa.get_landmarks_from_batch(image)
        if preds is not None and self.apply_smoothing:
            preds = self.smoother(preds)
        return preds

    def compute_mar(self, preds):
        # Compute mouth aspect ratio
        if preds is None:
            return np.nan
        preds = preds[:, 48:, :]
        # Compute the mouth aspect ratio
        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = np.linalg.norm(preds[:, 2] - preds[:, 10], axis=1)  # 51, 59
        B = np.linalg.norm(preds[:, 4] - preds[:, 8], axis=1)  # 53, 57

        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        C = np.linalg.norm(preds[:, 0] - preds[:, 6], axis=1)  # 49, 55

        # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar

    def compute_amplitude(self, preds):
        if preds is None:
            return np.nan
        preds = preds[:, self.stable_pt, :]  # (B, 2)
        i, j = furthest_away_pts(preds)
        if i is None or j is None:
            return np.nan
        return np.sqrt(np.sum((preds[i] - preds[j]) ** 2))

    def movement_std(self, preds):
        if preds is None:
            return np.nan
        preds = preds[:, self.stable_pt, :]
        distances = np.sqrt(np.sum(np.diff(preds, axis=0) ** 2, axis=1))
        return np.std(distances)

    def update(self, image, landmarks=None):
        if landmarks is not None:
            preds = landmarks
        else:
            preds = np.array(self.extract_landmarks(image))
        self.landmarks.append(preds)
        self.amplitude.append(self.compute_amplitude(preds))
        self.mov_std.append(self.movement_std(preds))
        self.mar.append(self.compute_mar(preds))
        # self.mar_std.append(np.nanstd(mar))

    def compute(self):
        # print(np.array(self.mar).shape)
        return (
            np.nanmean(self.amplitude),
            np.nanmean(self.mov_std),
            np.nanmean(self.mar),
            np.nanmean(np.std(self.mar, axis=1)),
        )

    def save_figures(self, path_dir):
        # Genearte boxplots of the amplitude and movement std
        Path(path_dir).mkdir(parents=True, exist_ok=True)
        boxplot_amplitude = sns.boxplot(data=self.amplitude).set_title("Amplitude")
        fig = boxplot_amplitude.get_figure()
        fig.savefig(os.path.join(path_dir, "amplitude.png"))
        fig.clf()
        boxplot_mov_std = sns.boxplot(data=self.mov_std).set_title("Movement std")
        fig = boxplot_mov_std.get_figure()
        fig.savefig(os.path.join(path_dir, "mov_std.png"))
        fig.clf()
        boxplot_mar_mean = sns.boxplot(data=np.mean(self.mar, axis=1)).set_title("MAR mean")
        fig = boxplot_mar_mean.get_figure()
        fig.savefig(os.path.join(path_dir, "mar_mean.png"))
        fig.clf()
        boxplot_mar_std = sns.boxplot(data=np.std(self.mar, axis=1)).set_title("MAR std")
        fig = boxplot_mar_std.get_figure()
        fig.savefig(os.path.join(path_dir, "mar_std.png"))
        fig.clf()

    def __sub__(self, other):
        mse_amplitude = np.mean((np.array(self.amplitude) - np.array(other.amplitude)) ** 2)
        mse_mar = np.mean((np.array(self.mar) - np.array(other.mar)) ** 2)
        try:
            mse_landmarks = np.mean((np.array(self.landmarks) - np.array(other.landmarks)) ** 2)
        except ValueError:
            print("Something went wrong with the landmarks")
            mse_landmarks = np.nan
        return mse_amplitude, mse_mar, mse_landmarks


if __name__ == "__main__":
    landmark_extractor = LandmarksExtractor()
    preds = landmark_extractor.extract_landmarks(
        "/vol/paramonos2/projects/antoni/code/generating_laugh/image_identity.png"
    )
    print(preds)
