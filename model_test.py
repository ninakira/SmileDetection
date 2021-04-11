import sys
import psutil
from keras.models import load_model

sys.path.append('../')
from data_access import load_celeba_test, load_genki_test


def test_model(model):
    celeba_test = load_celeba_test()
    genki_test = load_genki_test()

    celeba_loss, celeba_accuracy = model.evaluate(celeba_test)
    genki_loss, genki_accuracy = model.evaluate(genki_test)

    psutil.cpu_percent(interval=None)
    print('Celeba test accuracy :', celeba_accuracy)
    print('Genki test accuracy :', genki_accuracy)
    print('CPU usage percentage:', psutil.cpu_percent(interval=None))


def load_model_by_checkpoint(model_name, checkpoint_name):
    model_path = f'/home/aca1/code/SavedModels/{model_name}/checkpoints/{checkpoint_name}'
    return load_model(model_path)


def load_saved_model(model_name, fit_number):
    model_path = f'/home/aca1/code/SavedModels/{model_name}/SavedModel/{fit_number}'
    return load_model(model_path)
