import sys
from data_access import ImageDaoKeras
from training.model_training import KerasTrain
from models.VGG16 import VGG16
from models.BKNet import BKNet
from tensorflow.python.client import device_lib


def train_vgg_face(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path="../images/train")
    vgg_model = VGG16(dao_single_path).model
    trainer = KerasTrain(model=vgg_model,
                         name="Vgg_test_model",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=23,
                         epochs=3,
                         cp_freq=1)
    trainer.fit_model()


def train_bknet():
    dao_single_path = ImageDaoKeras(data_path="../images/train")
    bknet_model = BKNet(dao_single_path).model
    trainer = KerasTrain(model=bknet_model,
                         name="BKNet_test_model",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=23,
                         epochs=3,
                         cp_freq=1)
    trainer.fit_model()


def main(argv):
    print("Hey you! Smile! Arguments passed:", argv)
    # dao_single_path = ImageDaoKeras(data_path="images/train")
    dao_separate_paths = ImageDaoKeras(train_path="../images/train", validation_path="../images/validation")

    # train_vgg_face(argv[0])
    # train_bknet()


if __name__ == "__main__":
    print("Local devices", device_lib.list_local_devices())

    main(sys.argv[1:])
