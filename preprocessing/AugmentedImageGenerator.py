from keras.preprocessing.image import save_img
import numpy as np
import cv2


class AugmentedImageGenerator:
    def __init__(self,
                 dir_data,
                 dir_augmented_data,
                 generator_before_face_detection,
                 face_detector,
                 augmenter_after_face_detection):
        self.dir_data = dir_data
        self.dir_augmented_data = dir_augmented_data
        self.generator_before_face_detection = generator_before_face_detection
        self.face_detector = face_detector
        self.augmenter_after_face_detection = augmenter_after_face_detection
        self.start_index = 0
        self.n_saved_images = 0
        self.n_corrupted_images = 0

    def generate(self, start_index, n_images_to_generate):
        self.start_index = start_index
        for image, label in self.generator_before_face_detection \
                .flow_from_directory(self.dir_data,
                                     target_size=(178, 218),
                                     batch_size=1,
                                     class_mode='binary',
                                     shuffle=False):

            processed_image = self.process_image(image[0])
            if processed_image is None:
                self.n_corrupted_images += 1
                continue

            self.save_image(processed_image, label)

            if self.n_saved_images % 1000 == 0:
                print('Currently at image:', self.n_saved_images)
            if self.n_saved_images >= n_images_to_generate:
                print('Number of corrupted images:', self.n_corrupted_images)
                break

    def process_image(self, image):
        resulting_image = image

        if self.face_detector is not None:
            resulting_image = self.detect_face(image)

        if self.augmenter_after_face_detection is not None:
            try:
                resulting_image = self.augmenter_after_face_detection(resulting_image)
            except Exception as e:
                print(e)
                self.n_corrupted_images += 1

        return resulting_image

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
            name = f'{self.dir_augmented_data}/{converted_label}/{image_index}.jpg'
            save_img(name, resized)
            self.n_saved_images += 1
        except Exception as e:
            print(e)
            self.n_corrupted_images += 1
