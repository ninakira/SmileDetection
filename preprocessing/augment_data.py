from FaceDetector import FaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
import albumentations as A
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def augment_with_albumentations(image):
    transform = A.Compose([
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.IAASharpen(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.2, hue_shift_limit=10),
        A.GaussNoise(p=0.2),
    ])

    return transform(image=image)['image']


dir_data = "/data/original_celeba/celeba/validation"
dir_augmented_data = "/data/final_celeba/validation"

face_detector = FaceDetector()

datagen = ImageDataGenerator(rotation_range=15,
                             height_shift_range=0.1,
                             width_shift_range=0.1,
                             fill_mode="reflect",
                             zoom_range=0.1,
                             shear_range=0.1,
                             horizontal_flip=True)

generator = AugmentedImageGenerator(face_detector, dir_data, dir_augmented_data, augment_with_albumentations)

data = generator.generate(datagen,50000, 60000)

