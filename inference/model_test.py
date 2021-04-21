import sys
import psutil

sys.path.append('../')
from data_access import load_celeba_test, load_genki_test
from model_load import load_model_by_checkpoint, load_saved_model
from config import set_dynamic_memory_allocation


def test_model(model):
    celeba_test = load_celeba_test()
    genki_test = load_genki_test()

    celeba_loss, celeba_accuracy = model.evaluate(celeba_test)
    genki_loss, genki_accuracy = model.evaluate(genki_test)

    # psutil.cpu_percent(interval=None)
    print('Celeba test accuracy :', celeba_accuracy)
    print('Genki test accuracy :', genki_accuracy)
    print('CPU usage percentage:', psutil.cpu_percent(interval=1))

set_dynamic_memory_allocation()
model = load_model_by_checkpoint('MNasNet1', 'cp-0010-0.20.ckpt')
test_model(model)