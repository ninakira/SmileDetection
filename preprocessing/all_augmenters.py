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
    transform = A.ReplayCompose([
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.OneOf([
            A.IAASharpen(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.2, hue_shift_limit=10),
    ])
    tr = transform(image=image)
    image = tr['image']
    # print(tr['replay'])
    return image
