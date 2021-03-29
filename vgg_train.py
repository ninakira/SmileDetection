import sys
from data_access import ImageDaoKeras
from model_training import KerasTrain, get_exp_scheduler
from models.VGG16 import VGG16


def train_vgg(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path=data_path)

    vgg_model = VGG16(dao_single_path).model
    lr_scheduler = get_exp_scheduler(decay_step=1000)

    trainer = KerasTrain(model=vgg_model,
                         name="VGG_1000ep_genki_modified",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         batch_size=23,
                         epochs=1000,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def main(argv):
    assert len(argv) > 0
    data_path = argv[0]
    print("Training VGG on data from", data_path)
    train_vgg(data_path)


if __name__ == "__main__":
    main(sys.argv[1:])
