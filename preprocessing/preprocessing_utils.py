import os
import re
import zipfile
from pathlib import Path
from shutil import copyfile, move
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2


def move_or_copy_files(source, destination, operation="copy", n_images_to_operate=None):
    operator = move if operation == "move" else copyfile
    operated_image_count = 0
    for item in os.listdir(source):
        operated_image_count += 1
        source_filename = os.path.join(source, item)
        destination_filename = os.path.join(destination, item)
        operator(source_filename, destination_filename)
        if n_images_to_operate is None \
                or operated_image_count <= n_images_to_operate:
            continue


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

def sample_n_imgs(n):
    files = np.random.choice(os.listdir('/data/celeba/final_celeba/train/0'), size = n)
    print(files.shape)
    for f in files:
        print(f)
        image = cv2.imread(f'/data/celeba/final_celeba/train/0/{f}')
        print(image)
        plt.imsave(f'/data/celeba/final_celeba/sample/{f}', image)

def remove_pics():
    files = os.listdir('/data/noface')
    for f in files:
        if f[-3:] == 'jpg':
            os.remove(f'/data/noface/{f}')
