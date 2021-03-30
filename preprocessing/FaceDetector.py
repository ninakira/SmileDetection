import cv2

PATH = 'preprocessing/face_detection_models/face_detection_models/haarcascade_frontalcatface.xml'


class FaceDetector:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(PATH)

    def detect_faces(self, image):
        cropped_faces = []
        faces = self.face_detector.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        for (x, y, w, h) in faces:
            cropped_faces.append(image[y:y+h, x:x+w])

        if len(cropped_faces) == 0:
            return
        elif len(cropped_faces[0]) == 0:
            return
        return cropped_faces
