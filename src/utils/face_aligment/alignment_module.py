"""
Class to perform face alignment of a given image
"""
import numpy as np
import skimage
from src.utils.face_aligment.landmarks_extractor import LandmarksExtractor
from src.utils.face_aligment.face_processor import FaceProcessor


class FaceAligment:
    def __init__(
        self,
        mean_face_path,
        scale=1.0,
        device="cuda",
        fps=25,
        offset=None,
        out_size=None,
        square_it=True,
        reference=None,
    ):
        mean_face = scale * np.load(mean_face_path)
        self.landmark_extractor = LandmarksExtractor(device=device, fps=fps)

        height_border = 0
        width_border = 0
        if offset is not None:
            mean_face = FaceProcessor.offset_mean_face(mean_face, offset_percentage=offset[:2])
            height_border += offset[1] + offset[2]
            width_border += 2 * offset[0]
        if out_size is not None:
            crop_height = out_size[0]
            crop_width = out_size[1]
        elif reference is not None:
            reference = skimage.io.imread(reference)
            crop_height = reference.shape[0]
            crop_width = reference.shape[1]
        else:
            face_width, face_height = FaceProcessor.get_width_height(mean_face)
            crop_height = int(face_height * (1 + height_border))
            crop_width = int(face_width * (1 + width_border))

        if square_it:
            crop_height = crop_width = max(crop_height, crop_width)

        self.face_processor = FaceProcessor(
            cuda=device == "cuda", mean_face=mean_face, ref_img=reference, img_size=(crop_height, crop_width)
        )

    def process(self, image):
        # image: either a path to an image or a numpy array (H, W, C)
        if isinstance(image, str):
            image = skimage.io.imread(image)
        landmarks = self.landmark_extractor.extract_landmarks(image)
        if isinstance(landmarks, list):
            landmarks = landmarks[0]
        warped_img = self.face_processor.process_image(image, landmarks=landmarks)
        return warped_img
