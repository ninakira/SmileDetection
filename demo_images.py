import cv2
import glob
import numpy as np
import tensorflow as tf
from face_detector import HaarCascadeDetector, DlibHAGDetector, MTCNNDetector
from preprocessor import BasicPreprocessor
from config import get_label_text


class ImagesDemo:
    def __init__(self,
                 detector=None,
                 model=None,
                 preprocessor=None):
        self.model = model
        self.detector = detector
        self.preprocessor = preprocessor

    def process_images(self, file_path=None, scale=True):
        assert file_path is not None

        filenames = glob.glob(file_path + '/*.jpg')
        images = [cv2.imread(img) for img in filenames]

        for frame in images:
            faces = self.detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (71, 53, 6), 2)  # draw rectangle to main image

                detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                processed_img = self.preprocessor.get_preprocessed_img(detected_face, scale=scale)

                img = np.expand_dims(processed_img, axis=0)
                prediction = self.model.predict(img)[0, :]
                label = 1 if prediction[0] > 0 else 0

                cv2.putText(frame,
                            get_label_text(label),
                            (int(x + w - 100), int(y - 12)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7, (255, 255, 255), 1)

            cv2.imshow('Images Smile Detection Demo', cv2.resize(frame, (288, 268)))
            cv2.waitKey()

        cv2.destroyAllWindows()


haar = HaarCascadeDetector()
dlib = DlibHAGDetector()
mtcnn = MTCNNDetector()

proc = BasicPreprocessor(size=(128, 128))

model = tf.keras.models.load_model('tf_logs/Mobilenet3/SavedModel/1')

camera = ImagesDemo(model=model, detector=mtcnn, preprocessor=proc)
camera.process_images('demo_data/genki_sample', scale=False)
