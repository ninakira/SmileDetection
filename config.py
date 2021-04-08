import tensorflow as tf
DATA_PATH = "/data/"


def get_train_valid_paths(dir_name):
    return f"{DATA_PATH}{dir_name}/train", f"{DATA_PATH}{dir_name}/validation"


def set_dynamic_memory_allocation():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
