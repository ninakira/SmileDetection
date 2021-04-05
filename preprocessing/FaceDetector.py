import sys

sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        cropped_faces = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.01, 4)
        for (x, y, w, h) in faces:
            cropped_faces.append(image[y:y + h, x:x + w])

        if len(cropped_faces) == 0:
            return
        elif len(cropped_faces[0]) == 0:
            return
        else:
            return cropped_faces
