import albumentations as A
from keras.preprocessing.image import ImageDataGenerator

generator_before_face_detection = ImageDataGenerator(rotation_range=15,
                                                     height_shift_range=0.1,
                                                     width_shift_range=0.1,
                                                     fill_mode="reflect",
                                                     zoom_range=0.1,
                                                     shear_range=0.1,
                                                     horizontal_flip=True)

generator_no_augment = ImageDataGenerator()


def augmenter_after_face_detection(image):
    transform = A.Compose([
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.OneOf([
            # A.IAASharpen(),
            A.RandomBrightnessContrast(brightness_limit=0.05),
        ], p=1,
        # A.HueSaturationValue(p=1, hue_shift_limit=),
        # A.GaussNoise(p=0.5),
    ])
    
    return transform(image=image)['image']
