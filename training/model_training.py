import datetime
import os
import tensorflow as tf


class KerasTrain:
    def __init__(self, model=None,
                 name=None,
                 description=None,
                 train_data=None,
                 valid_data=None,
                 save_path="/home/aca1/code/SavedModels/",
                 with_cp_save=True,
                 cp_dir="checkpoints",
                 with_tensorboard=True,
                 tb_dir="tf_logs",
                 tb_hist_freq=10):
        self.model = model
        self.name = name
        self.description = description
        self.train_data = train_data
        self.valid_data = valid_data

        self.save_path = save_path + self.name + "/"
        self.with_cp_save = with_cp_save
        self.cp_dir = self.save_path + cp_dir

        self.with_tensorboard = with_tensorboard
        self.tb_dir = self.save_path + tb_dir
        self.tb_hist_freq = tb_hist_freq

        self.histories = []
        self.current_fit = 0

    def compile_model(self,
                      optimizer,
                      loss=None,
                      from_logits=True,
                      metrics=["accuracy"]):
        if loss is None:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits)

        self.model.compile(optimizer, loss, metrics)

    def fit_model(self,
                  epochs,
                  with_early_stop=False,
                  early_stop_patience=2,
                  initial_epoch=0,
                  save_weights_only=False):
        callbacks = []
        if self.with_tensorboard:
            callbacks.append(self.__get_tb_callback())
        if self.with_cp_save:
            callbacks.append(self.__get_cp_callback(save_weights_only))
        if with_early_stop:
            callbacks.append(self.__get_early_stop_callback(early_stop_patience))

        history = self.model.fit(
            self.train_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            validation_data=self.valid_data
        )

        self.histories.append(history)
        self.__save_model(self.current_fit)
        self.current_fit += 1

    def save_model(self):
        self.__save_model(self.current_fit)
        self.current_fit += 1

    def __save_model(self, fit_n):
        self.model.save(self.save_path + "SavedModel/{}".format(fit_n))

    def __get_early_stop_callback(self, early_stop_patience):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=early_stop_patience,
                                                restore_best_weights=True)

    def __get_cp_callback(self, save_weights_only=False):
        checkpoint_name = 'cp-{epoch:04d}.ckpt'
        checkpoint_path = f'{self.cp_dir}/{self.current_fit}/{checkpoint_name}'

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=save_weights_only,
            save_freq='epoch')

    def __get_tb_callback(self):
        logdir = os.path.join(f"{self.tb_dir}/{self.current_fit}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return tf.keras.callbacks.TensorBoard(logdir, histogram_freq=self.tb_hist_freq)

    def get_tensorlog_path(self):
        return self.tb_dir

    def get_history(self):
        return self.histories


def get_exp_scheduler(initial_learning_rate=0.05, decay_rate=0.96, decay_steps=100):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True)


def get_polymonial_scheduler(starter_learning_rate=0.1,
                             end_learning_rate=0.001,
                             decay_steps=10000):
    return tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)
