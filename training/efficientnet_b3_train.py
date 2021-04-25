import sys
import tensorflow as tf
from model_training import KerasTrain, get_polymonial_scheduler

sys.path.append('../')
import models.EfficientNet
from inference.model_load import load_saved_model
from data_access import ImageDaoKerasBigData, TRAIN_PATH, VALIDATION_PATH
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
        efficient_net = models.EfficientNet.EfficientNetV3()
        self.model = efficient_net.model
        self.base = efficient_net.base
        self.set_trainer(self.model, name)

        ep1 = frozen_epochs // 2
        self.train_frozen(ep1, frozen_lr)
        self.train_frozen(frozen_epochs - ep1, frozen_lr*0.1)

        epf1 = fine_tune_epochs // 2
        self.fine_tune(epf1, fine_tune_lr)
        self.fine_tune(fine_tune_epochs - epf1, fine_tune_lr*0.1)

    def train_frozen(self, epochs, lr):
        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs, with_early_stop=True, early_stop_patience=3, save_weights_only=False)

    def fine_tune(self, epochs, lr):
        for layer in self.model.layers:
            layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs, with_early_stop=True, early_stop_patience=3, save_weights_only=False)

    def train_saved_model(self, name, fit, epochs, lr, frozen=True):
        reconstructed_model = load_saved_model(name, fit)
        self.set_trainer(reconstructed_model, name)
        if not frozen:
            for layer in self.model.layers:
                layer.trainable = True

        optimizer = tf.keras.optimizers.Adam(lr)
        self.trainer.compile_model(optimizer)
        self.trainer.fit_model(epochs)


IMG_SIZE = (224, 224)

set_dynamic_memory_allocation()
dao = ImageDaoKerasBigData(train_path=TRAIN_PATH, validation_path=VALIDATION_PATH, img_size=IMG_SIZE)
celeba_train, celeba_validation = dao.load_data()

efficient_net3_trainer = EfficientNetTrainer(celeba_train, celeba_validation)
efficient_net3_trainer.train_new_model(name="EfficientNet_b3_celeba_1",
                                      frozen_epochs=15,
                                      frozen_lr=1e-2,
                                      fine_tune_epochs=20,
                                      fine_tune_lr=1e-4)

# efficient_net3_trainer.train_saved_model('EfficientNet_b3_celeba', 0, 20, 1e-5, False)


# test_model(model, img_size=IMG_SIZE, model_name='EfficientNet_b3_celeba')

