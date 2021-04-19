import cv2


class Preprocessor:
    def get_preprocessed_img(self, img) -> list:
        """Provide implementation for necessary preprocessing."""
        pass


class BasicPreprocessor(Preprocessor):
    def __init__(self, size=(128, 128)):
        self.size = size

    def get_preprocessed_img(self, img=None) -> list:
        assert img is not None

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        img = img / 255
        return img

