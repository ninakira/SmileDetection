import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
import models.MobileNetV3
from data_access import load_data
from config import set_dynamic_memory_allocation


TRAIN_PATH = "/data/final_celeba/train"
VALIDATION_PATH = "/data/final_celeba/validation"
IMG_SIZE = (128, 128)
BATCH_SIZE = 128


class MobileNetTrainer:
    def __init__(self, train_dataset, validation_dataset):
        self.model = None
        self.base = None
        self.trainer = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def set_trainer(self, model, name):
        self.trainer = KerasTrain(model,
                                  name,
                                  train_data=self.train_dataset,
                                  valid_data=self.validation_dataset)

    def train_new_model(self,
                        name,
                        frozen_epochs,
                        frozen_lr,
                        unfreeze_at,
                        fine_tune_epochs,
                        fine_tune_lr):
        mobilenet = models.MobileNetV3.MobileNetV3()
        self.model = mobilenet.define_model()
        self.base = mobilenet.get_base()
        self.set_trainer(self.model, name)

        self.train_frozen(frozen_epochs, frozen_lr)
        self.fine_tune(fine_tune_epochs, fine_tune_lr, unfreeze_at)

    def train_frozen(self, total_epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(total_epochs)

    def fine_tune(self, total_epochs, lr, fine_tune_at):
        self.base.trainable = True

        for layer in self.base.layers[:fine_tune_at]:
            layer.trainable = False

        histories = self.trainer.get_history()
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_modelfit_model(total_epochs,
                                        initial_epoch=histories[-1].epoch[-1])

    def train_saved_model(self, path, total_epochs, lr):
        reconstructed_model = tf.keras.models.load_model(path)
        self.set_trainer(reconstructed_model, "Reconstructed")

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(total_epochs)


set_dynamic_memory_allocation()
celeba_train, celeba_validation = load_data(TRAIN_PATH, VALIDATION_PATH, IMG_SIZE, BATCH_SIZE)

mobilenet_trainer = MobileNetTrainer(celeba_train, celeba_validation)
mobilenet_trainer.train_new_model(name="MobileNetV3_added_layer",
                                  frozen_epochs=2,
                                  frozen_lr=1e-4,
                                  unfreeze_at=50,
                                  fine_tune_epochs=3,
                                  fine_tune_lr=1e-5)
