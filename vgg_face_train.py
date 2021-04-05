import sys
import tensorflow as tf
from data_access import ImageDaoKerasBigData
from model_training import KerasTrain, get_polymonial_scheduler
from models.VGG_Face import VGGFace
from config import get_train_valid_paths


def train_vgg_face(train_path=None, valid_path=None):
    assert train_path is not None
    assert valid_path is not None

    dao_single_path = ImageDaoKerasBigData(train_path="images/train", validation_path="images/test")

    vgg_model = VGGFace().model
    lr_scheduler = get_polymonial_scheduler()

    trainer = KerasTrain(model=vgg_model,
                         name="VGGFace_400ep_celeba_aug",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=5974,
                         epochs=400,
                         with_early_stop=True,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def reload_vgg_face_and_train():
    dao_single_path = ImageDaoKerasBigData(train_path="../../../../data/augmented_celeba/train",
                                    validation_path="../../../../data/augmented_celeba/validation")

    loaded_model = tf.keras.models.load_model('../SavedModels/VGGFace_200ep_celeba_aug/SavedModel/0/')
    lr_scheduler = get_polymonial_scheduler()

    trainer = KerasTrain(model=loaded_model,
                         name="VGGFace_400ep_celeba_aug",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=5974,
                         epochs=400,
                         with_early_stop=True,
                         lr_scheduler=lr_scheduler)
    trainer.fit_model()


def main(argv):
    data_path = argv[0] if len(argv) > 0 else "augmented_celeba"
    print("Training VGG Face on data from", data_path)
    train_path, valid_path = get_train_valid_paths(data_path)
    train_vgg_face(train_path=train_path, valid_path=valid_path)


if __name__ == "__main__":
    main(sys.argv[1:])
