import sys
from NoFaceDetector import NoFaceDetector
from NoFaceImageGenerator import NoFaceImageGenerator
from all_augmenters import generator_no_augment
sys.path.append('../')
from config import set_dynamic_memory_allocation


DIR_DATA = "/data/noface/test/"
DIR_PROCESSED_DATA = "/data/noface/augmented_test"
START_INDEX = 0
N_IMAGES = 5000


def generate_images(dir_data,
                    dir_processed_data,
                    image_name_start_index,
                    n_images_to_generate):

    no_face_detector = NoFaceDetector()
    generator = NoFaceImageGenerator(dir_data,
                                        dir_processed_data,
                                        generator_no_augment,
                                        no_face_detector,
                                        generator_no_augment,
                                        shuffle=True
                                        )

    generator.generate(image_name_start_index, n_images_to_generate)


set_dynamic_memory_allocation()
generate_images(DIR_DATA, DIR_PROCESSED_DATA, START_INDEX, N_IMAGES)