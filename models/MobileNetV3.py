import tensorflow as tf

IMG_SHAPE = (128, 128, 3)


class MobileNetV3:
    def __init__(self):
        self.preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        self.base_model = tf.keras.applications.MobileNetV3Large(input_shape=IMG_SHAPE,
                                                                 include_top=False,
                                                                 weights='imagenet')
        self.base_model.trainable = False

    def define_model(self):
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def get_base(self):
        return self.base_model
