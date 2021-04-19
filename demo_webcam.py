import cv2
from face_detector import HaarCascadeDetector, DlibDetector, MTCNNDetector
from preprocessor import BasicPreprocessor


def get_label_text(label=None):
    assert label is not None
    switcher = {
        0: "No Smile",
        1: "Smile!",
    }
    return switcher.get(label, "Invalid label")


class CameraDemo:
    def __init__(self,
                 detector=None,
                 model=None,
                 preprocessor=None):
        self.model = model
        self.detector = detector
        self.preprocessor = preprocessor

    def start_camera(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            faces = self.detector.detect_faces(frame)

            for x, y, w, h in faces:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (138, 88, 14), 2)  # draw rectangle to main image
                detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face

                processed_img = self.preprocessor.get_preprocessed_img(detected_face)
                # prediction = self.model.predict(processed_img)[0, :]

                prediction = 0
                cv2.putText(frame,
                            get_label_text(prediction),
                            (int(x + w - 100), int(y - 12)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (255, 255, 255), 2)

            cv2.imshow('Smile Detection Demo', frame)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
            else:
                pass

        # Close openCV video capture imshow
        cap.release()
        cv2.destroyAllWindows()


haar = HaarCascadeDetector()
dlib = DlibDetector()
mtcnn = MTCNNDetector()

proc = BasicPreprocessor(size=(128, 128))

camera = CameraDemo(detector=dlib, preprocessor=proc)
camera.start_camera()
