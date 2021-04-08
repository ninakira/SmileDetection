from data_access import ImageDaoKerasBigData
from models.MobileNetV3 import MobileNetV3
from model_training import KerasTrain, get_exp_scheduler
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

TRAIN_PATH = "/data/final_celeba/train"
VALIDATION_PATH = "/data/final_celeba/validation"
IMG_SIZE = (128, 128)
BATCH_SIZE = 128


def set_dynamic_memory_alocation():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_data():
    train_dataset = image_dataset_from_directory(TRAIN_PATH,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    validation_dataset = image_dataset_from_directory(VALIDATION_PATH,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset

def train_mobilenetv3():
    train_dataset, validation_dataset = load_data()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    mobilenet = MobileNetV3()
    model = mobilenet.define_model()

    trainer = KerasTrain(model=model,
                         name="MobileNetV3_trial",
                         train_data=train_dataset,
                         valid_data=validation_dataset,
                         optimizer=optimizer,
                         epochs=1)

    trainer.fit_model()

    base = mobilenet.get_base()
    base.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 1

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    histories = trainer.get_history()
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    trainer.compile_model(fine_tune_optimizer)
    trainer.fit_model(initial_epoch=histories[-1].epoch[-1], total_epochs=2)


def load_previous_model():
    train_dataset, validation_dataset = load_data()
    reconstructed_model = tf.keras.models.load_model("/home/aca1/code/SavedModels/SavedModel/MobileNetV3_trial/1/")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    trainer = KerasTrain(model=reconstructed_model,
                         name="MobileNetV3_trial_v2",
                         train_data=train_dataset,
                         valid_data=validation_dataset,
                         optimizer=optimizer,
                         epochs=1)
    trainer.fit_model(total_epochs=3)


set_dynamic_memory_alocation()
load_previous_model()
