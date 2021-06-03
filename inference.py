import numpy as np
import time
import tensorflow as tf
from face_detector import HaarCascadeDetector, DlibHAGDetector, MTCNNDetector, Yolo3Tiny, MPFaceDetector


def compute_time(model=None, detector=None, size=(128, 128)):
    det = np.random.rand(350, 350, 3).astype(np.uint8)

    # blob = cv2.dnn.blobFromImage(det, 1 / 255, (179, 192),
    #                              [0, 0, 0], 1, crop=False)
    inputs = np.random.rand(1, size[0], size[1], 3).astype(np.uint8)

    # warm-up
    for kk in range(0, 1000):
        ll = 1

    i = 0
    time_spent = []
    print("#"*100)
    tic = time.perf_counter()
    while i < 1000:
        start_time = time.time()

        _ = detector.detect_faces(img=det, blob=None)
        _ = model(inputs)
        i += 1
        time_spent.append(time.time() - start_time)

    tac = time.perf_counter()

    avg = 'Avg execution time (ms): {:.5f}'.format(np.mean(time_spent))
    max = 'Max execution time (ms): {:.5f}'.format(np.max(time_spent))
    std = 'Std execution time (ms): {:.5f}'.format(np.std(time_spent))
    start = f"Start: {tic}"
    end = f"End: {tac}"
    total = f"Total time: {tac-tic}"
    fps = f"FPS: {1000/(tac-tic)}"

    results = [avg, max, std, start, end, total, fps]

    out_f = open("inference_results_all.txt", "w")
    for res in results:
        print(res)
        out_f.write(res)
        out_f.write("\n")

    print("#"*100)
    print("\n")
    out_f.write("\n")
    out_f.write("\n")

    out_f.close()


def compute_detector_time(detector, size=(128, 128)):
    det = np.random.rand(350, 350, 3).astype(np.uint8)

    # blob = cv2.dnn.blobFromImage(det, 1 / 255, (179, 192),
    #                              [0, 0, 0], 1, crop=False)

    for kk in range(0, 1000):
        ll = 1

    i = 0
    time_spent = []
    print("#"*100)
    tic = time.perf_counter()
    while i < 1000:
        start_time = time.time()
        _ = detector.detect_faces(img=det, blob=None)
        i += 1
        time_spent.append(time.time() - start_time)

    tac = time.perf_counter()

    avg = 'Avg execution time (ms): {:.5f}'.format(np.mean(time_spent))
    max = 'Max execution time (ms): {:.5f}'.format(np.max(time_spent))
    std = 'Std execution time (ms): {:.5f}'.format(np.std(time_spent))
    start = f"Start: {tic}"
    end = f"End: {tac}"
    total = f"Total time: {tac-tic}"
    fps = f"FPS: {1000/(tac-tic)}"

    results = [avg, max, std, start, end, total, fps]

    out_f = open("inference_results_all.txt", "w")
    # out_f.write(name)
    for res in results:
        print(res)
        out_f.write(res)
        out_f.write("\n")

    print("#"*100)
    print("\n")
    out_f.write("\n")
    out_f.write("\n")

    out_f.close()


# haar = HaarCascadeDetector()
# dlib = DlibHAGDetector()
# mtcnn = MTCNNDetector()
# yolo = Yolo3Tiny()



path = 'saved_model/USmile/SavedModel/0'
mp = MPFaceDetector()

model = tf.keras.models.load_model(path)
compute_time(model, mp)

