import sys
from FaceDetector import FaceDetector
from AugmentedImageGenerator import AugmentedImageGenerator
from all_augmenters import generator_no_augment

sys.path.append('../')
from config import set_dynamic_memory_allocation


DIR_DATA = "/data/genki/original_genki"
DIR_PROCESSED_DATA = "/data/genki/face_detected_genki"
START_INDEX = 1
N_IMAGES = 4000


def generate_images(dir_data,
                    dir_processed_data,
                    image_name_start_index,
                    n_images_to_generate):

    face_detector = FaceDetector()
    generator = AugmentedImageGenerator(dir_data,
                                        dir_processed_data,
                                        generator_no_augment,
                                        face_detector,
                                        None)

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_PROCESSED_DATA, START_INDEX, N_IMAGES)
