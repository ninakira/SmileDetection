import sys
from keras.models import load_model

sys.path.append('../')
from data_access import load_celeba_test, load_genki_test

celeba_test = load_celeba_test()
genki_test = load_genki_test()


def test_model(checkpoint_path):
    model = load_model(checkpoint_path)
    celeba_loss, celeba_accuracy = model.evaluate(celeba_test)
    genki_loss, genki_accuracy = model.evaluate(genki_test)

    print('Celeba test accuracy :', celeba_accuracy)
    print('Test accuracy :', genki_accuracy)
