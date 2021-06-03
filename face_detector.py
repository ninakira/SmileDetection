import cv2
import dlib
import numpy as np
import math
from mtcnn.mtcnn import MTCNN
import mediapipe as mp
from demo_util import *


class FaceDetector:
    def detect_faces(self, img=None, blob=None) -> list:
        """Provide implementation for detecting faces"""
        pass


class HaarCascadeDetector(FaceDetector):
    def __init__(self):
        self.detector = cv2.CascadeClassifier('demo_data/Haarcascades/haarcascade_frontalface_default.xml')

    def detect_faces(self, img=None, blob=None) -> list:
        return self.detector.detectMultiScale(img, 1.2, 8)


class DlibHAGDetector(FaceDetector):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, img=None, blob=None) -> list:
        faces = self.detector(img, 1)
        return list(map(lambda face: (face.left(),
                                      face.top(),
                                      face.right() - face.left(),
                                      face.bottom() - face.top()), faces))


class MTCNNDetector(FaceDetector):
    def __init__(self):
        super().__init__()
        self.detector = MTCNN()

    def detect_faces(self, img=None, blob=None) -> list:
        faces = self.detector.detect_faces(img)
        return list(map(lambda face: face['box'], faces))


class MPFaceDetector(FaceDetector):
    def __init__(self):
        self._mp_face_detection = mp.solutions.face_detection
        self._face_detection = self._mp_face_detection.FaceDetection(min_detection_confidence=0.4)
        self._mp_drawing = mp.solutions.drawing_utils

    def detect_faces(self, img=None, blob=None) -> list:
        res = []
        with self._mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    # self._mp_drawing.draw_detection(img, detection)
                    # print(detection)
                    xmin = detection.location_data.relative_bounding_box.xmin
                    ymin = detection.location_data.relative_bounding_box.ymin
                    width = detection.location_data.relative_bounding_box.width
                    height = detection.location_data.relative_bounding_box.height

                    # xmin = xmin - width * 0.1
                    # ymin = ymin - height * 0.1
                    # width = width + width * 0.2
                    # height = height + height * 0.2

                    left = int(xmin * (img.shape[1] - 1))
                    top = int(ymin * (img.shape[0] - 1))
                    width = int(width * (img.shape[1] - 1))
                    height = int(height * (img.shape[0] - 1))

                    res.append([left, top, width, height])
        return res


yolo_model_path = './yolo3_tiny/face-yolov3-tiny.cfg'
# Download weights from 'https://drive.google.com/file/d/1xG0QhxwWtyB17PKfifxHV9FPSck2z1Bk/view'
yolo_weights_path = './yolo3_tiny/face-yolov3-tiny_41000.weights'


class Yolo3Tiny(FaceDetector):
    def __init__(self):
        # Give the configuration and weight files for the model and load the network
        # using them.
        net = cv2.dnn.readNetFromDarknet(yolo_model_path, yolo_weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.detector = net

    # Get the names of the output layers
    def get_outputs_names(self):
        # Get the names of all the layers in the network
        layers_names = self.detector.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected
        # outputs
        return [layers_names[i[0] - 1] for i in self.detector.getUnconnectedOutLayers()]

    def post_process(self, frame, outs, conf_threshold, nms_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only
        # the ones with high confidence scores. Assign the box's class label as the
        # class with the highest score.
        confidences = []
        boxes = []
        final_boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                                   nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            final_boxes.append(box)
        return final_boxes

    def detect_faces(self, img=None, blob=None) -> list:
        self.detector.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.detector.forward(self.get_outputs_names())
        faces = self.post_process(img, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        return faces
