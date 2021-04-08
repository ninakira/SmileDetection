from mtcnn.mtcnn import MTCNN


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image):
        cropped_faces = []

        faces = self.detector.detect_faces(image)
        for result in faces:
            x, y, w, h = result['box']
            cropped_faces.append(image[y:y + h, x:x + w])

        if len(cropped_faces) == 0:
            return
        elif len(cropped_faces[0]) == 0:
            return
        else:
            return cropped_faces
