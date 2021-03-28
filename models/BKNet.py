import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential


class BKNet:

    gaussian_std = 0.02
    l2_val = 0.02
    momentum = 0.9

    def __init__(self,
                 dataset=None):
        """
        BKNet model from https://www.researchgate.net/publication/321259348_Facial_smile_detection_using_convolutional_neural_networks

        :param dataset: ImageDaoKeras or ImageDaoCustom object
        """
        assert dataset is not None

        self.model = Sequential([
            self.data_augment(dataset.IMG_HEIGHT, dataset.IMG_WIDTH),

            layers.experimental.preprocessing.Rescaling(1. / 255),

            self.conv2D(32),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            self.conv2D(32),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            self.conv2D(64),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            self.conv2D(64),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            self.conv2D(128),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            self.conv2D(128),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            self.conv2D(256),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            self.conv2D(256),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            self.conv2D(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            layers.Flatten(),

            self.dense(256),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            self.dense(256),
            layers.BatchNormalization(momentum=self.momentum),
            layers.Activation('relu'),
            layers.Dropout(0.5),

            layers.Dense(1)
        ])


    def data_augment(self, height, width):
        return keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(height, width, 3)),
                layers.experimental.preprocessing.RandomRotation(0.15),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )


    def conv2D(self, n_filters):
        return layers.Conv2D(n_filters, 3, padding='same',
                             kernel_initializer=initializers.RandomNormal(stddev=self.gaussian_std),
                             bias_initializer=initializers.Ones(),
                             kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_val))


    def dense(self, units):
        return layers.Dense(units,
                            kernel_initializer=initializers.RandomNormal(stddev=self.gaussian_std),
                            bias_initializer=initializers.Ones(),
                            kernel_regularizer=tf.keras.regularizers.L2(l2=self.l2_val))
