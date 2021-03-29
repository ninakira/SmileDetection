import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class VGG16:
    def __init__(self,
                 dataset=None):
        """
        VGG16 model with pretrained weight from imagenet.

        :param dataset: ImageDaoKeras or ImageDaoCustom object
        """
        assert dataset is not None

        base_model = tf.keras.applications.VGG16(input_shape=(dataset.IMG_HEIGHT, dataset.IMG_WIDTH, 3),
                                                 include_top=False,
                                                 weights='imagenet')

        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(dataset.IMG_HEIGHT, dataset.IMG_WIDTH, 3)),
                layers.experimental.preprocessing.RandomRotation(0.15),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)

        # freeze vgg
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(dataset.IMG_HEIGHT, dataset.IMG_WIDTH, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)

        self.model = tf.keras.Model(inputs, outputs)

