import sys
from data_access import ImageDaoKeras
from model_training import KerasTrain, get_exp_scheduler
from models.VGG_Face import VGGFace


def train_vgg_face(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path=data_path)

    vgg_model = VGGFace().model
    lr_scheduler = get_exp_scheduler(decay_step=200)

    trainer = KerasTrain(model=vgg_model,
                         name="VGGFace_200ep_celeba_aug",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=4481,
                         epochs=200,
                         with_cp_save=False,
                         with_tensorboard=False,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def main(argv):
    assert len(argv) > 0
    data_path = argv[0]
    print("Training VGG Face on data from", data_path)
    train_vgg_face(data_path)


if __name__ == "__main__":
    main(sys.argv[1:])
