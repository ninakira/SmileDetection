import cv2
import numpy as np
import tensorflow as tf
from face_detector import HaarCascadeDetector, DlibHAGDetector, MTCNNDetector
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

    def process_video(self, file_path=None, scale=True):
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
                    processed_img = self.preprocessor.get_preprocessed_img(detected_face, scale)

                    img = np.expand_dims(processed_img, axis=0)
                    prediction = self.model.predict(img)[0, :]
                    label = 1 if prediction[0] > 0 else 0

                    cv2.putText(frame,
                                get_label_text(label),
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
dlib = DlibHAGDetector()
mtcnn = MTCNNDetector()

proc = BasicPreprocessor(size=(128, 128))

model = tf.keras.models.load_model('tf_logs/Mobilenet3/SavedModel/1')

camera = VideoDemo(model=model, detector=haar, preprocessor=proc)
# camera.process_video('demo_data/Friends.mp4', False)
camera.process_video('demo_data/smiling.mp4', False)
