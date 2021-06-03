import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
# from config import SAVED_MODELS_PATH
from models.MobileNetV3 import MobileNetV3
# from data_access import load_mixed
# from config import set_dynamic_memory_allocation


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
        mobilenet = MobileNetV3()
        self.model = mobilenet.define_model()
        print(self.model.summary())
        self.base = mobilenet.get_base()
        print(self.base.layers)
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

    def train_saved_model(self, new_name, full_path, epochs, lr):
        reconstructed_model = tf.keras.models.load_model(full_path)
        self.set_trainer(reconstructed_model, new_name)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

#
# set_dynamic_memory_allocation()
# mixed_train, mixed_validation = load_mixed()
#
# SAVED_MODEL_PATH = '/home/aca1/code/SavedModels/Mobilenet_Small4_retrain3/checkpoints/0/cp-0020-0.14.ckpt'
# mobilenet_trainer = MobileNetTrainer(mixed_train, mixed_validation)
#
# mobilenet_trainer.train_saved_model('Mobilenet_Small4_retrain4', SAVED_MODEL_PATH, 20, 5e-6)


mobilenet = MobileNetV3()
model = mobilenet.define_model()
print(model.summary())
