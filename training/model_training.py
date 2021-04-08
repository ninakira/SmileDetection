import datetime
import os
import tensorflow as tf


class KerasTrain:
    def __init__(self, model=None,
                 name=None,
                 description=None,
                 train_data=None,
                 valid_data=None,
                 iters=23,
                 from_logits=True,
                 epochs=400,
                 current_epoch=0,
                 lr=0.001,
                 lr_scheduler=None,
                 optimizer=None,
                 loss=None,
                 metrics=["accuracy"],
                 with_early_stop=False,
                 early_stop_patience=8,
                 save_path="home/aca1/code/SavedModels/",
                 with_cp_save=True,
                 cp_freq=10,
                 cp_dir="checkpoints/",
                 with_tensorboard=True,
                 tb_dir="tf_logs/",
                 tb_hist_freq=10):
        self.model = model
        self.name = name
        self.description = description
        self.train_data = train_data
        self.valid_data = valid_data
        self.iters = iters
        self.from_logits = from_logits
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.with_early_stop = with_early_stop
        self.early_stop_patience = early_stop_patience

        self.save_path = save_path + self.name + "/"
        self.with_cp_save = with_cp_save
        self.cp_freq = cp_freq
        self.cp_dir = self.save_path + cp_dir

        self.with_tensorboard = with_tensorboard
        self.tb_dir = self.save_path + tb_dir
        self.tb_hist_freq = tb_hist_freq

        self.histories = []

        self.compile_model()
        self.current_fit = 0

    def compile_model(self, optimizer=None):
        optimizer = self.__get_optimizer(optimizer)

        loss = self.loss if self.loss is not None \
            else tf.keras.losses.BinaryCrossentropy(from_logits=self.from_logits)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=self.metrics)

    def fit_model(self, initial_epoch=0, total_epochs=None):
        callbacks = []
        if self.with_tensorboard:
            callbacks.append(self.__get_tb_callback())
        if self.with_cp_save:
            callbacks.append(self.__get_cp_callback())
        if self.with_early_stop:
            callbacks.append(self.__get_early_stop_callback())

        if total_epochs is not None:
            self.epochs = total_epochs

        history = self.model.fit(
            self.train_data,
            validation_data=self.valid_data,
            epochs=self.epochs,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
        )

        self.histories.append(history)
        self.__save_model(self.current_fit)
        self.current_fit += 1

    def __get_optimizer(self, optimizer_arg):
        if optimizer_arg is not None:
            return optimizer_arg
        if self.optimizer is not None:
            return self.optimizer
        if self.lr_scheduler is not None:
            return tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler)
        return tf.keras.optimizers.Adam(lr=self.lr)

    def __save_model(self, fit_n):
        self.model.save(self.save_path + "SavedModel/{}".format(fit_n))

    def __get_early_stop_callback(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=self.early_stop_patience,
                                                restore_best_weights=True)

    def __get_cp_callback(self):
        checkpoint_path = self.cp_dir + "cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.model.save_weights(checkpoint_path.format(epoch=self.current_epoch))

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            verbose=1,
            save_weights_only=True,
            save_freq=self.cp_freq * self.iters)

    def __get_tb_callback(self):
        logdir = os.path.join(self.tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return tf.keras.callbacks.TensorBoard(logdir, histogram_freq=self.tb_hist_freq)

    def get_tensorlog_path(self):
        return self.tb_dir

    def get_history(self):
        return self.histories

def get_exp_scheduler(initial_learning_rate=0.05, decay_rt=0.96, decay_step=100):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_step,
        decay_rate=decay_rt,
        staircase=True)


def get_polymonial_scheduler(starter_learning_rate=0.1,
                             end_learning_rate=0.001,
                             decay_steps=10000):
    return tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)
