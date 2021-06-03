import cv2
import numpy as np


class Preprocessor:
    def get_preprocessed_img(self, img, scale) -> list:
        """Provide implementation for necessary preprocessing."""
        pass


class BasicPreprocessor(Preprocessor):
    def __init__(self, size=(128, 128, 3)):
        self.size = size

    def get_preprocessed_img(self, img=None, scale=False) -> list:
        assert img is not None

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)

        img = np.expand_dims(img, axis=0)

        if scale:
            img = img / 255

        return img
