import sys
import tensorflow as tf
from tensorflow.keras import layers
from model_training import KerasTrain

sys.path.append('../')
import models.EfficientNet
from data_access import load_celeba
from config import set_dynamic_memory_allocation


class EfficientNetTrainer:
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
                        fine_tune_epochs,
                        fine_tune_lr):
        efficient_net = models.EfficientNet.EfficientNetV0()
        self.model = efficient_net.model
        self.base = efficient_net.base
        self.set_trainer(self.model, name)

        self.model.load_weights("/home/aca1/code/SavedModels/EfficientNet_with_finetuning_final_celeba/checkpoints/cp-0020-0.38.ckpt")
        # self.trainer.save_model()

        # self.train_frozen(frozen_epochs, frozen_lr)
        self.fine_tune(fine_tune_epochs, fine_tune_lr)

    def train_frozen(self, epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs, with_early_stop=True, save_weights_only=True)

    def fine_tune(self, epochs, lr):
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen

        for layer in self.model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        histories = self.trainer.get_history()
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs, initial_epoch=histories[-1].epoch[-1],
                               save_weights_only=True, with_early_stop=True,)

    def train_saved_model(self, name, path, epochs, lr):
        reconstructed_model = tf.keras.models.load_model(path)
        self.set_trainer(reconstructed_model, name)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)


IMG_SIZE = (224, 224)

set_dynamic_memory_allocation()
celeba_train, celeba_validation = load_celeba(img_size=IMG_SIZE)

efficient_net_trainer = EfficientNetTrainer(celeba_train, celeba_validation)
efficient_net_trainer.train_new_model(name="EfficientNet_only_finetuning_final_celeba",
                                      frozen_epochs=25,
                                      frozen_lr=1e-4,
                                      fine_tune_epochs=25,
                                      fine_tune_lr=1e-5)
