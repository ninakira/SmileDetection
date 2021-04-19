import cv2
import dlib
from mtcnn.mtcnn import MTCNN


class FaceDetector:
    def detect_faces(self, img) -> list:
        """Provide implementation for detecting faces"""
        pass


class HaarCascadeDetector(FaceDetector):
    def __init__(self):
        self.detector = cv2.CascadeClassifier('demo_data/Haarcascades/haarcascade_frontalface_default.xml')

    def detect_faces(self, img) -> list:
        return self.detector.detectMultiScale(img, 1.3, 6)


class DlibDetector(FaceDetector):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, img) -> list:
        faces = self.detector(img, 1)
        return list(map(lambda face: (face.left(),
                                      face.top(),
                                      face.right() - face.left(),
                                      face.bottom() - face.top()), faces))


class MTCNNDetector(FaceDetector):
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, img) -> list:
        faces = self.detector.detect_faces(img)
        return list(map(lambda face: face['box'], faces))
