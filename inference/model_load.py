from keras.models import load_model
import sys

sys.path.append('../')
from config import SAVED_MODELS_PATH


def load_model_by_checkpoint(model_name, checkpoint_name):
    model_path = f'/home/aca1/code/SavedModels/{model_name}/checkpoints/{checkpoint_name}'
    return load_model(model_path)


def load_saved_model(model_name, fit_number):
    model_path = f'/home/aca1/code/SavedModels/{model_name}/SavedModel/{str(fit_number)}'
    return load_model(model_path)
