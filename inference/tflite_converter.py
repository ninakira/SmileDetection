import sys
import tensorflow as tf

sys.path.append('../')
from config import SAVED_MODELS_PATH, TFLITE_MODELS_PATH


def convert_and_save(saved_model_dir, model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open(f'{TFLITE_MODELS_PATH}/{model_name}.tflite', 'wb') as f:
        f.write(tflite_model)


model_dir = SAVED_MODELS_PATH + '/MNasNet1/checkpoints/cp-0010-0.20.ckpt'
model_name = 'mnasnet1'
convert_and_save(model_dir, model_name)
