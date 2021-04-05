import sys
import tensorflow as tf
from data_access import ImageDaoKeras
from model_training import KerasTrain, get_exp_scheduler, get_polymonial_scheduler
from models.VGG_Face import VGGFace


def train_vgg_face(data_path=None):
    dao_single_path = ImageDaoKeras(train_path="../../../../data/augmented_celeba/train",
                                    validation_path="../../../../data/augmented_celeba/validation")

    vgg_model = VGGFace().model
    lr_scheduler = get_exp_scheduler(decay_step=200)

    trainer = KerasTrain(model=vgg_model,
                         name="VGGFace_400ep_celeba_aug",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=5974,
                         epochs=200,
                         with_early_stop=True,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def reload_vgg_face_and_train():
    dao_single_path = ImageDaoKeras(train_path="../../../../data/augmented_celeba/train",
                                    validation_path="../../../../data/augmented_celeba/validation")

    loaded_model = tf.keras.models.load_model('../SavedModels/VGGFace_200ep_celeba_aug/SavedModel/0/saved_model.pb')
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
    # data_path = argv[0]
    # print("Training VGG Face on data from", data_path)
    train_vgg_face()


if __name__ == "__main__":
    main(sys.argv[1:])
