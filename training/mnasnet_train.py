import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
from models.MNasNet import MNasNet
from data_access import load_celeba
from config import set_dynamic_memory_allocation


class MNasNetTrainer:
    def __init__(self, train_dataset, validation_dataset):
        self.model = None
        self.trainer = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def set_trainer(self, model, name):
        self.trainer = KerasTrain(model,
                                  name,
                                  train_data=self.train_dataset,
                                  valid_data=self.validation_dataset)

    def train_model(self, name, epochs, lr=0.001):
        self.model = MNasNet()
        self.set_trainer(self.model, name)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

    def train_saved_model(self, name, path, epochs, lr):
        reconstructed_model = tf.keras.models.load_model(path)
        self.set_trainer(reconstructed_model, name)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)


set_dynamic_memory_allocation()
celeba_train, celeba_validation = load_celeba()

MNASNET_TRAINED_PATH = '/home/aca1/code/SavedModels/MNasNet1/checkpoints/cp-0011-0.24.ckpt'
mnasnet_trainer = MNasNetTrainer(celeba_train, celeba_validation)
mnasnet_trainer.train_saved_model(name="MNasNet1",path=MNASNET_TRAINED_PATH, epochs=10, lr=0.0001)




