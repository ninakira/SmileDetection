import sys
from FaceDetector import FaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
from all_augmenters import generator_before_face_detection, augmenter_after_face_detection

sys.path.append('../')
from config import set_dynamic_memory_allocation


DIR_DATA = "/data/celeba/original_celeba/celeba/train"
DIR_PROCESSED_DATA = "/data/celeba/non_face_detected_celeba/test copy 2"
START_INDEX = 1
N_IMAGES = 40


def generate_images(dir_data,
                    dir_processed_data,
                    image_name_start_index,
                    n_images_to_generate):

    face_detector = None
    generator = AugmentedImageGenerator(dir_data,
                                        dir_processed_data,
                                        generator_before_face_detection,
                                        face_detector,
                                        None, 
                                        shuffle=True)

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_PROCESSED_DATA, START_INDEX, N_IMAGES)
