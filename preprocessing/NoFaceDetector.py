from mtcnn.mtcnn import MTCNN
import cv2 

class NoFaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image):
        cropped_faces = []
        
        faces_box = self.detector.detect_faces(image)
        print("no_face_detect: ", type(faces_box))
        print("no_face_detect: ", faces_box)
        if(faces_box is not None and len(faces_box) != 0):
            for face_box in faces_box:
                x, y, w, h = face_box['box']
                color = (0, 0, 0)
                thickness = -1
                image = cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness)
                
        return image
        
        

        