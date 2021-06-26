import os
import re
import shutil
import zipfile
from pathlib import Path
from shutil import copyfile, move
from keras.preprocessing.image import ImageDataGenerator


def move_or_copy_files(source, destination, n_images_to_operate=None, operation="copy"):
    operator = move if operation == "move" else copyfile
    operated_image_count = 1
    for item in os.listdir(source):
        source_filename = os.path.join(source, item)
        destination_filename = os.path.join(destination, item)
        operator(source_filename, destination_filename)
        operated_image_count += 1
        if n_images_to_operate is None \
                or operated_image_count <= n_images_to_operate:
            continue
        else:
            break


def selective_move(source, destination, expression):
    for item in os.listdir(source):
        if re.search(expression, item) is None:
            continue
        source_filename = os.path.join(source, item)
        destination_filename = os.path.join(destination, item)
        move(source_filename, destination_filename)


def read_data_check(source, class_mode='binary', batch_size=128):
    data_generator = ImageDataGenerator()

    data_generator.flow_from_directory(source, class_mode=class_mode, batch_size=batch_size)


def extract_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        path = Path(zip_path)
        name = os.path.basename(path).split('.')[0]
        zip_ref.extractall(str(path.parent) + "/unzipped_" + name)


def clear_folder(path):
    folder_0 = os.path.join(path, '0')
    folder_1 = os.path.join(path, '1')
    if os.path.exists(folder_0):
        shutil.rmtree(folder_0)
    if os.path.exists(folder_1):
        shutil.rmtree(folder_1)
    os.makedirs(folder_0)
    os.makedirs(folder_1)


# source = '/data/affectnet/divided/train/1'
# destination = '/home/aca1/code/code_Lilit/AugPlayground/1'
# n_images_to_operate = 1000
# move_or_copy_files(source, destination, n_images_to_operate, "copy")