from data_access import ImageDaoKerasBigData
from models.MobileNetV3 import MobileNetV3
from model_training import KerasTrain
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

TRAIN_PATH = "/data/final_celeba/train"
VALIDATION_PATH = "/data/final_celeba/validation"
IMG_SIZE = (128, 128)
BATCH_SIZE = 128

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
    reconstructed_model = tf.keras.models.load_model("/home/aca1/code/SavedModels/MobileNetV3_trial/1")
    print(reconstructed_model)
    # reconstructed_model
    # train_dataset, validation_dataset = load_data()
    #
    # trainer = KerasTrain(model=reconstructed_model,
    #                      name="MobileNetV3_trial",
    #                      train_data=train_dataset,
    #                      valid_data=validation_dataset,
    #                      optimizer=optimizer,
    #                      epochs=1)
    #
    # trainer.fit_model()
    # reconstructed_model.fit(test_input, test_target)
load_previous_model()
