import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3


# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224


class EfficientNetV0:

    def __init__(self):
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        self.base = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

        # Freeze the pretrained weights
        self.base.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(self.base.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")


class EfficientNetV3:

    def __init__(self):
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        self.base = EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")

        self.base.trainable = False

        x = layers.GlobalAveragePooling2D(name="avg_pool")(self.base.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs, outputs, name="EfficientNetB3")
