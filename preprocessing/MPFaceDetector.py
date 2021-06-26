import cv2
import mediapipe as mp


class MPFaceDetector():
    def __init__(self):
        self._mp_face_detection = mp.solutions.face_detection
        self._face_detection = self._mp_face_detection.FaceDetection(min_detection_confidence=0.4)
        self._mp_drawing = mp.solutions.drawing_utils

    def detect_faces(self, img=None, blob=None) -> list:
        cropped_faces = []
        with self._mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    xmin = detection.location_data.relative_bounding_box.xmin
                    ymin = detection.location_data.relative_bounding_box.ymin
                    width = detection.location_data.relative_bounding_box.width
                    height = detection.location_data.relative_bounding_box.height

                    left = int(xmin * (img.shape[1] - 1))
                    top = int(ymin * (img.shape[0] - 1))
                    width = int(width * (img.shape[1] - 1))
                    height = int(height * (img.shape[0] - 1))

                    cropped_faces.append(img[top:top + height, left:left + width])

        if len(cropped_faces) == 0:
            return
        elif len(cropped_faces[0]) == 0:
            return
        else:
            print("Found faces MTCNN: ", len(cropped_faces))
            return cropped_faces[0]
