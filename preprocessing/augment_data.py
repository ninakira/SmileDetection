from FaceDetector import FaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
from config import set_dynamic_memory_allocation
from all_augmenters import generator_no_augment


DIR_DATA = "/data/original_celeba/celeba/test"
DIR_AUGMENTED_DATA = "/data/final_celeba/test"
START_INDEX = 0
N_IMAGES = 10000


def generate_images(dir_data,
                    dir_augmented_data,
                    image_name_start_index,
                    n_images_to_generate):

    face_detector = FaceDetector()
    generator = AugmentedImageGenerator(dir_data,
                                        dir_augmented_data,
                                        generator_no_augment,
                                        face_detector,
                                        None)

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_AUGMENTED_DATA, START_INDEX, N_IMAGES)
