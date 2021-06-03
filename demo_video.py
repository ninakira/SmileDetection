import sys
import cv2
from face_detector import HaarCascadeDetector, DlibHAGDetector, MTCNNDetector, Yolo3Tiny, MPFaceDetector
from preprocessor import BasicPreprocessor
from keras.models import load_model
import time
from demo_util import *


class VideoDemo:
    def __init__(self,
                 detector=None,
                 model=None,
                 preprocessor=None):
        self.model = model
        self.detector = detector
        self.preprocessor = preprocessor

    def process_video(self, file_path=None, freq=1):
        assert file_path is not None

        cap = cv2.VideoCapture(file_path)

        i = 0

        boxes = []
        preds = []
        text_num = ""

        out = cv2.VideoWriter('output2.mp4', -1, 20, (500, 282))

        tic = time.perf_counter()
        while True:
            i = i + 1
            ret, frame = cap.read()

            # Stop the program if reached end of video
            if not ret:
                print(f'[{i}] ==> Done processing!!!')
                cv2.waitKey(1000)
                break

            if frame is None:
                break

            if i % freq == 0:

                faces = self.detector.detect_faces(img=frame)

                text_num = f'number of faces detected:{len(faces)}'
                cv2.putText(frame, text_num, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
                boxes = []
                preds = []

                for face in faces:
                    boxes.append(face)
                    left, top, right, bottom = refined_box(face[0], face[1], face[2], face[3])

                    (x, y, w, h) = face
                    detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                        img = self.preprocessor.get_preprocessed_img(detected_face)

                        prediction = sigmoid(self.model.predict(img)[0, :][0])
                        preds.append(prediction)
                        draw_predict(frame, left, top, right, bottom, conf=prediction)

                print(f'[{i}] ==> # detected faces: {len(faces)}')
                print('#' * 60)

            else:
                for j in range(len(boxes)):
                    if len(preds) >= j+1:
                        cv2.putText(frame, text_num, (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1)
                        face = boxes[j]
                        left, top, right, bottom = refined_box(face[0], face[1], face[2], face[3])
                        draw_predict(frame, left, top, right, bottom, conf=preds[j])

            out.write(frame)
            cv2.imshow('Video Smile Detection Demo', frame)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
            else:
                pass

        tac = time.perf_counter()
        print(f"FRAME {i}; Total time: {tac - tic}")
        print(f"SPF: {(tac - tic) / i}")
        print(f"FPS: {i / (tac - tic)}")

        out.release()
        cap.release()
        cv2.destroyAllWindows()


haar = HaarCascadeDetector()
dlib = DlibHAGDetector()
mtcnn = MTCNNDetector()
yolo = Yolo3Tiny()
mp = MPFaceDetector()

proc = BasicPreprocessor(size=(130, 130))
detector = mp

# model = tf.keras.models.load_model('tf_logs/Mobilenet3/SavedModel/1')

# model_path = 'tf_logs/Mobilenet_Small1/checkpoints/1/1/cp-0019-0.21.ckpt'
# model = load_model(model_path)

model_path = './tf_logs/FINAL_Unet/SavedModel/SavedModel/0'
model = load_model(model_path)

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

camera = VideoDemo(model=model, detector=detector, preprocessor=proc)


camera.process_video('demo_data/laugh.mp4', 2)
# camera.process_video('demo_data/celeb.mp4', 2)
# camera.process_video('demo_data/beautiful.mp4', 1)
# camera.process_video('demo_data/beautiful.mp4', 2)
# camera.process_video('demo_data/beautiful.mp4', 2)
# camera.process_video('demo_data/dude_sm.mp4', 1)

