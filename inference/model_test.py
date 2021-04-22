import sys
import time

sys.path.append('../')
from data_access import load_celeba_test, load_genki_test
from inference.model_load import load_model_by_checkpoint, load_saved_model
from config import set_dynamic_memory_allocation


def test_model(model, img_size=(128, 128), model_name=None):
    celeba_test = load_celeba_test(img_size=img_size)
    genki_test = load_genki_test(img_size=img_size)

    tic = time.perf_counter()
    celeba_loss, celeba_accuracy = model.evaluate(celeba_test)
    toc = time.perf_counter()
    genki_loss, genki_accuracy = model.evaluate(genki_test)
    tac = time.perf_counter()

    # psutil.cpu_percent(interval=None)
    celeba = f'Celeba test accuracy :{celeba_accuracy}'
    celeba_time = f'Celeba time: {toc - tic:0.4f}'
    genki = f'Genki test accuracy : {genki_accuracy}'
    genki_time = f'Genki time: {tac - tic:0.4f}'
    # cpu = f'CPU usage percentage: {psutil.cpu_percent(interval=1)}'
    results = [celeba, celeba_time, genki, genki_time]

    out_f = open(f"/home/aca1/code/model_results/{model_name}_inference.txt", "w")
    for res in results:
        print(res)
        out_f.write(res)
        out_f.write("\n")
    out_f.close()


if __name__ == "__main__":
    set_dynamic_memory_allocation()
    model = load_model_by_checkpoint('MNasNet1', 'cp-0010-0.20.ckpt')
    test_model(model)
