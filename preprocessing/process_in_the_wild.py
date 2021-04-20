import sys
from AugmentedImageGenerator import AugmentedImageGenerator
from all_augmenters import generator_before_face_detection, augmenter_after_face_detection

sys.path.append('../')
from config import set_dynamic_memory_allocation


DIR_DATA = "/data/celeba/original_celeba/celeba/validation"
DIR_PROCESSED_DATA = "/data/celeba/non_face_detected_celeba/validation"
START_INDEX = 0
N_IMAGES = 10000


def generate_images(dir_data,
                    dir_processed_data,
                    image_name_start_index,
                    n_images_to_generate):

    face_detector = None
    generator = AugmentedImageGenerator(dir_data,
                                        dir_processed_data,
                                        generator_before_face_detection,
                                        face_detector,
                                        augmenter_after_face_detection, 
                                        shuffle=True)

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_PROCESSED_DATA, START_INDEX, N_IMAGES)
