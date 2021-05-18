import os
import tensorflow as tf
import zipfile
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory

AUTOTUNE = tf.data.AUTOTUNE


class ImageDaoKeras:
    def __init__(self, data_path=None,
                 train_path=None,
                 validation_path=None,
                 img_size=(224, 224),
                 valid_split=0.25,
                 batch_size=128,
                 color_format="rgb"):
        self.img_size = img_size
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
            image_size=self.img_size,
            shuffle=True,
            seed=42,
            validation_split=valid_split,
            subset=subset_type,
            interpolation="bilinear",
            follow_links=False,
        )

    def load_data(self):
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.valid_dataset = self.valid_dataset.prefetch(buffer_size=AUTOTUNE)
        return self.train_dataset, self.valid_dataset


class ImageDaoKerasBigData:
    def __init__(self,
                 train_path=None,
                 validation_path=None,
                 img_size=(128, 128),
                 valid_split=0.25,
                 batch_size=128,
                 color_format="rgb"):
        assert train_path is not None
        assert validation_path is not None

        self.img_size = img_size
        self.valid_split = valid_split
        self.color_format = color_format
        self.batch_size = batch_size

        self.train_dataset = self.load_data_keras(train_path)
        self.valid_dataset = self.load_data_keras(validation_path)

    def load_data_keras(self, data_path):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        return datagen.flow_from_directory(
            data_path,
            class_mode="binary",
            color_mode=self.color_format,
            batch_size=self.batch_size,
            target_size=self.img_size,
            shuffle=True,
            seed=42,
            interpolation="bilinear",
            follow_links=False,
        )

    def load_data(self):
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.valid_dataset = self.valid_dataset.prefetch(buffer_size=AUTOTUNE)
        return self.train_dataset, self.valid_dataset


def merged_dataset(data_path1, data_path2):
    data1 = ImageDaoKeras(train_path=f'{data_path1}/train',
                          validation_path=f'{data_path1}/validation')
    data2 = ImageDaoKeras(train_path=f'{data_path2}/train',
                          validation_path=f'{data_path2}/validation')

    train_data = MergedDataGen(data1.train_dataset.as_numpy_iterator(),
                               data2.train_dataset.as_numpy_iterator())
    valid_data = MergedDataGen(data1.valid_dataset.as_numpy_iterator(),
                               data2.valid_dataset.as_numpy_iterator())

    return train_data.generate(), valid_data.generate()


class MergedDataGen:
    def __init__(self, *gens):
        self.gens = gens
        self.toggle = 0

    def generate(self):
        while True:
            try:
                i = self.toggle
                self.toggle = (self.toggle + 1) % len(self.gens)
                yield next(self.gens[i])
            except StopIteration:
                break

    def __len__(self):
        return sum([len(g) for g in self.gens])


class DataExtractor:
    def __init__(self, data_path):
        self.data_path = data_path

        self.extract_zip(data_path)

    def extract_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            path = Path(zip_path)
            name = os.path.basename(path).split('.')[0]
            zip_ref.extractall(str(path.parent) + "/unzipped_" + name)

TRAIN_PATH = "/data/celeba/final_celeba/train"
VALIDATION_PATH = "/data/celeba/final_celeba/validation"

TEST_PATH_CELEBA = "/data/celeba/final_celeba/test"
TEST_PATH_GENKI = "/data/genki/face_detected_genki"

IMG_SIZE = (128, 128)


def load_celeba(batch_size=128, img_size=IMG_SIZE):
    train_dataset = image_dataset_from_directory(TRAIN_PATH,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 image_size=img_size)

    validation_dataset = image_dataset_from_directory(VALIDATION_PATH,
                                                      shuffle=True,
                                                      batch_size=batch_size,
                                                      image_size=img_size)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset


def load_celeba_test(batch_size=128, img_size=(128, 128)):
    test_dataset = image_dataset_from_directory(TEST_PATH_CELEBA,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                image_size=img_size)

    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return test_dataset


def load_genki_test(batch_size=128, img_size=(128, 128)):
    test_dataset = image_dataset_from_directory(TEST_PATH_GENKI,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                image_size=img_size)

    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return test_dataset


#
# train_data, valid_data = merged_dataset('test-images/test1', 'test-images/test2')
#
