import numpy as np
from keras.models import load_model
import time
import sys
sys.path.append('/home/aca1/code/code_Lilit/SmileDetection')
from config import SAVED_MODELS_PATH


def computeTime(model):
    inputs = np.random.rand(1, 128, 128, 3)

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        _ = model(inputs)

        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))


model_path = SAVED_MODELS_PATH + '/Mobilenet_Small4_retrain3/checkpoints/0/cp-0018-0.14.ckpt'
model = load_model(model_path)
computeTime(model)
