from data_access import ImageDaoKerasBigData
from models.MobileNetV3 import MobileNetV3
from model_training import KerasTrain
import tensorflow as tf

TRAIN_PATH = "/data/final_celeba/train"
VALIDATION_PATH = "/data/final_celeba/validation"


def train_mobilenetv3():
    dao = ImageDaoKerasBigData(train_path=TRAIN_PATH, validation_path=VALIDATION_PATH)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    mobilenet = MobileNetV3()
    model = mobilenet.define_model()

    trainer = KerasTrain(model=model,
                         name="MobileNetV3_trial",
                         train_data=dao.train_dataset,
                         valid_data=dao.valid_dataset,
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


train_mobilenetv3()
