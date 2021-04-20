import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
from models import Xception
from data_access import load_celeba_3classes_train
from model_test import load_model_by_checkpoint, test_model
from config import set_dynamic_memory_allocation


class XceptionTrainer:
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
        xception = Xception.Xception()
        self.model = xception.define_model()
        print(self.model.summary())
        self.base = xception.get_base()
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
celeba_train, celeba_val  = load_celeba_3classes_train()

xception_trainer = XceptionTrainer(celeba_train, celeba_val)
xception_trainer.train_new_model(name="Xception_frozen",
                                  frozen_epochs=20,
                                  frozen_lr=1e-4,
                                  fine_tune_at=150,
                                  fine_tune_epochs=0,
                                  fine_tune_lr=1e-5)


