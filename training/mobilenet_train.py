import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
import models.MobileNetV3
from data_access import load_celeba
from config import set_dynamic_memory_allocation


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
                        fine_tune_at,
                        fine_tune_epochs,
                        fine_tune_lr):
        mobilenet = models.MobileNetV3.MobileNetV3()
        self.model = mobilenet.define_model()
        self.base = mobilenet.get_base()
        self.set_trainer(self.model, name)

        self.train_frozen(frozen_epochs, frozen_lr)
        self.fine_tune(fine_tune_epochs, fine_tune_lr, fine_tune_at)

    def train_frozen(self, epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

    def fine_tune(self, epochs, lr, fine_tune_at):
        self.base.trainable = True

        for layer in self.base.layers[:fine_tune_at]:
            layer.trainable = False

        histories = self.trainer.get_history()
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs, initial_epoch=histories[-1].epoch[-1])

    def train_saved_model(self, name, path, epochs, lr):
        reconstructed_model = tf.keras.models.load_model(path)
        self.set_trainer(reconstructed_model, name)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)


set_dynamic_memory_allocation()
celeba_train, celeba_validation = load_celeba()

mobilenet_trainer = MobileNetTrainer(celeba_train, celeba_validation)
mobilenet_trainer.train_new_model(name="Mobilenet_flatten_layer",
                                  frozen_epochs=2,
                                  frozen_lr=1e-4,
                                  fine_tune_at=50,
                                  fine_tune_epochs=10,
                                  fine_tune_lr=1e-5)
