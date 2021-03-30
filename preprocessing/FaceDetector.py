import dlib

PATH = '/home/aca1/code/SmileDetection/preprocessing/face_detection_models/mmod_human_face_detector.dat'


class FaceDetector:

    def __init__(self, mode='train'):
        self.mode = mode

        if self.mode == 'train':
            self.face_detector = dlib.cnn_face_detection_model_v1(PATH)
        elif self.mode == 'test':
            self.face_detector = dlib.get_frontal_face_detector()
        else:
            raise ValueError("Mode should be either 'train' or 'test'")

    def detect_faces(self, image):
        cropped_faces = []
        faces = self.face_detector(image, 1)

        for (i, rect) in enumerate(faces):
            if self.mode == 'train':
                rect = rect.rect
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
            cropped_faces.append(image[y1:y2, x1:x2])

        if len(cropped_faces) == 0:
            return
        elif len(cropped_faces[0]) == 0:
            return
        return cropped_faces
