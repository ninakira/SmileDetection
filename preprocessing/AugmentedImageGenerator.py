from keras.preprocessing.image import save_img
import numpy as np
import cv2


class AugmentedImageGenerator:
    def __init__(self, face_detector, dir_data, dir_aug_data, augment_after_detection=None):
        self.face_detector = face_detector
        self.dir_data = dir_data
        self.dir_aug_data = dir_aug_data
        self.augment_after_detection = augment_after_detection
        self.corrupted_image_count = 0
        self.n_saved_images = 0
        self.start_index = 0

    def generate(self, datagen, start_index, n_pics):
        self.start_index = start_index
        for image, label in datagen.flow_from_directory(self.dir_data,
                                                        target_size=(178, 218),
                                                        batch_size=1,
                                                        class_mode='binary'):
            if int(label[0]) == 1: 
                continue
            aug_image = self.augment_image(image[0])
            if aug_image is None:
                self.corrupted_image_count += 1
                continue

            self.save_image(aug_image, label)

            if self.n_saved_images % 100 == 0:
                print('Currently at image:', self.n_saved_images)
            if self.n_saved_images >= n_pics:
                print('Number of corrupted images:', self.corrupted_image_count)
                break

    def augment_image(self, image):
        detected_face = self.detect_face(image)
        if self.augment_after_detection and type(detected_face) == np.ndarray:
            try:
                return self.augment_after_detection(detected_face)
            except:
                pass

        return detected_face

    def detect_face(self, image):
        converted_image = np.array(image, dtype='uint8')
        faces = self.face_detector.detect_faces(converted_image)
        if faces is None:
            return

        return faces[0]

    def save_image(self, face, label):
        try:
            resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
            converted_label = str(int(label[0]))
            image_index = str(self.n_saved_images + self.start_index)
            name = f'{self.dir_aug_data}/{converted_label}/{image_index}.jpg'
            save_img(name, resized)
            self.n_saved_images += 1
        except Exception as e:
            print(e)
            self.corrupted_image_count += 1
