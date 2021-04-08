from FaceDetector import FaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
from config import set_dynamic_memory_allocation
from all_augmenters import generator_before_face_detection, augmenter_after_face_detection


DIR_DATA = "/data/original_celeba/celeba/validation"
DIR_AUGMENTED_DATA = "/data/final_celeba/validation"
START_INDEX = 50000
N_IMAGES = 10000


def generate_images(dir_data,
                    dir_augmented_data,
                    image_name_start_index,
                    n_images_to_generate):

    face_detector = FaceDetector()
    generator = AugmentedImageGenerator(dir_data,
                                        dir_augmented_data,
                                        generator_before_face_detection,
                                        face_detector,
                                        augmenter_after_face_detection)

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_AUGMENTED_DATA, START_INDEX, N_IMAGES)