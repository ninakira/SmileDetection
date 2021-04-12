import sys
import tensorflow as tf
from model_training import KerasTrain

sys.path.append('../')
import models.VGG_Face
from data_access import load_celeba
from config import set_dynamic_memory_allocation


class VGGTrainer:
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
        vgg = models.VGG_Face.VGGFace()
        self.model = vgg.model
        self.set_trainer(self.model, name)

        self.train_frozen(frozen_epochs, frozen_lr)
        self.fine_tune(fine_tune_epochs, fine_tune_lr)

    def train_frozen(self, epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)

    def fine_tune(self, epochs, lr):
        for layer in self.model.layers[:-1]:
            layer.trainable = True

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


IMG_SIZE = (224, 224)

set_dynamic_memory_allocation()
celeba_train, celeba_validation = load_celeba(img_size=IMG_SIZE)

vgg_trainer = VGGTrainer(celeba_train, celeba_validation)
vgg_trainer.train_new_model(name="VGG_face_pretrained_withfinetuning_final_celeba",
                            frozen_epochs=20,
                            frozen_lr=1e-4,
                            fine_tune_epochs=100,
                            fine_tune_lr=1e-5)
