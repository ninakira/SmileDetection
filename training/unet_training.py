import sys
import tensorflow as tf
from model_training import KerasTrain
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import SpatialDropout2D


sys.path.append('/home/aca1/code/code_Lilit/SmileDetection')
from data_access import load_mixed, load_celeba
from config import set_dynamic_memory_allocation


def define_unet():
    input = Input(shape=(128, 128, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(input)
    x = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(192, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = SpatialDropout2D(0.3)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.3)(x)
    output = Dense(1)(x)
    model = tf.keras.Model(input, output, name="Unet")
    return model


set_dynamic_memory_allocation()
mixed_train, mixed_validation = load_celeba()

model = define_unet()
trainer = KerasTrain(model,
                     name="Unet_192_celeba",
                     train_data=mixed_train,
                     valid_data=mixed_validation)


optimizer = tf.keras.optimizers.Adam(0.0001)
trainer.compile_model(optimizer)
trainer.fit_model(100, with_early_stop=True, early_stop_patience=3)