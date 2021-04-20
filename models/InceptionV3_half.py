import tensorflow as tf

IMG_SHAPE = (96, 96, 3)


class InceptionV3_half:
    def __init__(self):
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        self.base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                                 include_top=False,
                                                                 weights='imagenet')
        self.base_model = self.get_model_part(0,196, self.base_model, 'InceptionV3_AUX')
        self.base_model.trainable = False

    def get_model_part(self, start_ind, end_ind, model, name):
        if(start_ind != 0):
            layer_0 = model.layers[start_ind-1]
            inputs = tf.keras.Input(shape=layer_0.output.shape)
            model.layers[start_ind](inputs)
        else:
            inputs = model.layers[start_ind].output
        outputs = model.layers[end_ind].output
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    def define_model(self):
        '''
        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(rate = 0.3)(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate = 0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        '''

        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = self.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate = 0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def get_base(self):
        return self.base_model
