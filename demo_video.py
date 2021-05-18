import sys
import cv2
import numpy as np
import tensorflow as tf
from face_detector import HaarCascadeDetector, DlibHAGDetector, MTCNNDetector, Yolo3Tiny, MPFaceDetector
from preprocessor import BasicPreprocessor
from config import get_label_text
from keras.models import load_model
import time
from demo_util import *

FREQ = 1


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

        i = 0
        tic = time.perf_counter()

        faces = []
        while True:
            i = i + 1
            ret, frame = cap.read()

            cap.set(cv2.CAP_PROP_FPS, 1)

            # Stop the program if reached end of video
            if not ret:
                print('[i] ==> Done processing!!!')
                cv2.waitKey(1000)
                break

            if frame is None:
                break

            print("FRAME", frame.shape)

            # Re-size video to a smaller size to improve face detection speed
            frame = cv2.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3)))

            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)

            if i % FREQ == 0:
                faces = self.detector.detect_faces(img=frame, blob=blob)

                print('#' * 60)
                print('[i] ==> # detected faces: {}'.format(len(faces)))

                tac = time.perf_counter()
                print(f"FRAME {i}; time: {tac - tic}")
                print(f"SPF: {(tac - tic) / i}")
                print(f"FPS: {i / (tac - tic)}")

            info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]

            for (j, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (j * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_RED, 2)

            for face in faces:
                left, top, right, bottom = refined_box(face[0], face[1], face[2], face[3])
                draw_predict(frame, left, top, right, bottom)

                # (x, y, w, h) = face
                # detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                # processed_img = self.preprocessor.get_preprocessed_img(detected_face, scale)
                #
                # img = np.expand_dims(processed_img, axis=0)
                # prediction = self.model.predict(img)[0, :]
                # label = 1 if prediction[0] > 0 else 0

            # cv2.imshow('Video Smile Detection Demo', cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0]))))
            cv2.imshow('Video Smile Detection Demo', frame)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
            else:
                pass

        cap.release()
        cv2.destroyAllWindows()


haar = HaarCascadeDetector()
dlib = DlibHAGDetector()
mtcnn = MTCNNDetector()
yolo = Yolo3Tiny()
mp = MPFaceDetector()

proc = BasicPreprocessor(size=(128, 128))
detector = mp

# model = tf.keras.models.load_model('tf_logs/Mobilenet3/SavedModel/1')

# model_path = 'tf_logs/Mobilenet_Small1/checkpoints/1/1/cp-0019-0.21.ckpt'
# model = load_model(model_path)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", sys.argv[1:])
args = sys.argv[1:]
if len(args) > 0:
    detector_arg = args[0]
    if detector_arg == 'haar':
        detector = haar
    elif detector_arg == 'dlib':
        detector = dlib
    elif detector_arg == 'mtcnn':
        detector = mtcnn
    elif detector_arg == 'yolo':
        detector = yolo
    elif detector_arg == 'mediapipe':
        detector = mp

camera = VideoDemo(model=None, detector=detector, preprocessor=proc)
# camera.process_video('demo_data/Friends.mp4', False)

camera.process_video('demo_data/smiling.mp4', False)
