import sys
from data_access import ImageDaoKeras
from model_training import KerasTrain
from models.VGG_Face import VGG_Face
from models.BKNet import BKNet


def train_vgg_face(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path="images/train")
    vgg_model = VGG_Face(dao_single_path).model
    trainer = KerasTrain(model=vgg_model,
                         name="Vgg_test_model",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         batch_size=23,
                         epochs=3,
                         cp_freq=1)
    trainer.fit_model()


def train_bknet():
    dao_single_path = ImageDaoKeras(data_path="images/train")
    bknet_model = BKNet(dao_single_path).model
    trainer = KerasTrain(model=bknet_model,
                         name="BKNet_test_model",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         batch_size=23,
                         epochs=3,
                         cp_freq=1)
    trainer.fit_model()


def main(argv):
    print("Hey you! Smile!", argv)
    # dao_single_path = ImageDaoKeras(data_path="images/train")
    # dao_separate_paths = ImageDaoKeras(train_path="images/train", validation_path="images/test")

    train_vgg_face(argv[0])
    # train_bknet()


if __name__ == "__main__":
    main(sys.argv[1:])
