import cv2
from face_detector import HaarCascadeDetector, DlibDetector, MTCNNDetector
from preprocessor import BasicPreprocessor
from config import get_label_text


class VideoDemo:
    def __init__(self,
                 detector=None,
                 model=None,
                 preprocessor=None):
        self.model = model
        self.detector = detector
        self.preprocessor = preprocessor

    def process_video(self, file_path=None):
        assert file_path is not None

        cap = cv2.VideoCapture(file_path)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (320, 180))  # Re-size video to a smaller size to improve face detection speed

            faces = self.detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                if w > 13:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (71, 53, 6), 1)  # draw rectangle to main image

                    detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                    processed_img = self.preprocessor.get_preprocessed_img(detected_face)

                    # prediction = self.model.predict(processed_img)[0, :]
                    prediction = 0

                    cv2.putText(frame, get_label_text(prediction),
                                (int(x + w + 5), int(y - 12)),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.5, (255, 255, 255), 1)

                    # connect face and text
                    cv2.line(frame, (int((x + x + w) / 2), y + 5), (x + w, y - 10), (71, 53, 6), 1)

            cv2.imshow('Video Smile Detection Demo', cv2.resize(frame, (640, 360)))

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
            else:
                pass

        cap.release()
        cv2.destroyAllWindows()


haar = HaarCascadeDetector()
dlib = DlibDetector()
mtcnn = MTCNNDetector()

proc = BasicPreprocessor(size=(128, 128))

camera = VideoDemo(detector=dlib, preprocessor=proc)
camera.process_video('demo_data/Friends.mp4')
# camera.process_video('demo_data/Friends_long.mp4')
