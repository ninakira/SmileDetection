# import numpy as np
import os
import tensorflow as tf
# import cv2
import zipfile
from pathlib import Path
# from sklearn.model_selection import train_test_split

class ImageDaoKeras:
    def __init__(self, data_path=None,
                 train_path=None,
                 validation_path=None,
                 height=224,
                 width=224,
                 valid_split=0.25,
                 batch_size=128,
                 color_format="rgb"):
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.valid_split = valid_split
        self.color_format = color_format
        self.batch_size = batch_size

        if data_path is not None:
            self.train_dataset = self.load_data_keras(data_path, "training", valid_split=self.valid_split)
            self.valid_dataset = self.load_data_keras(data_path, "validation", valid_split=self.valid_split)
        else:
            self.train_dataset = self.load_data_keras(train_path)
            self.valid_dataset = self.load_data_keras(validation_path)


    def load_data_keras(self, data_path, subset_type=None, valid_split=None):
        return tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            labels="inferred",
            label_mode="binary",
            color_mode=self.color_format,
            batch_size=self.batch_size,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            shuffle=True,
            seed=42,
            validation_split=valid_split,
            subset=subset_type,
            interpolation="bilinear",
            follow_links=False,
        )

# class ImageDaoCustom:
#     def __init__(self, data_path, height=224, width=224, valid_split=0.25, batch_size=128):
#         self.data_path = data_path
#         self.IMG_HEIGHT = height
#         self.IMG_WIDTH = width
#         self.valid_split = valid_split
#         self.batch_size = batch_size
#
#         self.load_data()
#
#
#     def load_data(self):
#         self.data, self.labels = self.load_data_from_path(self.data_path)
#
#         x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=self.valid_split, random_state=42)
#
#         self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#         self.train_dataset = self.train_dataset.batch(self.batch_size)
#
#         self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#         self.test_dataset = self.test_dataset.batch(self.batch_size)
#
#
#     def load_data_from_path(self, img_folder):
#         img_data_array = []
#         class_name = []
#
#         for dir1 in os.listdir(img_folder):
#             for file in os.listdir(os.path.join(img_folder, dir1)):
#                 image_path = os.path.join(img_folder, dir1, file)
#                 image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
#                 image = cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)
#                 image = np.array(image)
#                 image = image.astype('float32')
#                 image /= 255
#                 img_data_array.append(image)
#                 class_name.append(dir1)
#
#         return img_data_array, class_name


class DataExtractor:
    def __init__(self, data_path):
        self.data_path = data_path

        self.extract_zip(data_path)


    def extract_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            path = Path(zip_path)
            name = os.path.basename(path).split('.')[0]
            zip_ref.extractall(str(path.parent) + "/unzipped_" + name)
