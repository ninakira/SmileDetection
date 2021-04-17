import sys
from NoFaceDetector import NoFaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
from all_augmenters import generator_before_face_detection, augmenter_after_face_detection

sys.path.append('../')
from config import set_dynamic_memory_allocation


DIR_DATA = "/data/noface/lsun/"
DIR_PROCESSED_DATA = "/data/noface/augmented_lsun"
START_INDEX = 400001
N_IMAGES = 650000


def generate_images(dir_data,
                    dir_processed_data,
                    image_name_start_index,
                    n_images_to_generate):

    no_face_detector = NoFaceDetector()
    generator = AugmentedImageGenerator(dir_data,
                                        dir_processed_data,
                                        generator_before_face_detection,
                                        no_face_detector,
                                        augmenter_after_face_detection,
                                        shuffle=True
                                        )

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_PROCESSED_DATA, START_INDEX, N_IMAGES)