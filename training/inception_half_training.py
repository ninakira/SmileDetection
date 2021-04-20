import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
from models import InceptionV3_half
from data_access import load_celeba_3classes_train
from model_test import load_model_by_checkpoint, test_model
from config import set_dynamic_memory_allocation


class InceptionHalfTrainer:
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
        inception = InceptionV3_half.InceptionV3_half()
        self.model = inception.define_model()
        print(self.model.summary())
        self.base = inception.get_base()
        self.set_trainer(self.model, name)

        self.train_frozen(frozen_epochs, frozen_lr)
        self.fine_tune(fine_tune_epochs, fine_tune_lr, fine_tune_at)

    def train_frozen(self, epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

    def fine_tune(self, epochs, lr, fine_tune_at):
        self.model.trainable = True

        for layer in self.model.layers[:fine_tune_at]:
            layer.trainable = False

       
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

    def train_saved_model(self, name, path , epochs, lr ):
        self.model = tf.keras.models.load_model(path)

        self.set_trainer(self.model, name)

        self.fine_tune(epochs,lr, 133)


set_dynamic_memory_allocation()
celeba_train, celeba_val  = load_celeba_3classes_train()

inception_trainer = InceptionHalfTrainer(celeba_train, celeba_val)
inception_trainer.train_new_model(  name="InceptionV3_half_frozen_small2",
                                    frozen_epochs = 10,
                                    frozen_lr = 1e-4,
                                    fine_tune_at = 133,
                                    fine_tune_epochs = 10,
                                    fine_tune_lr = 1e-4)


