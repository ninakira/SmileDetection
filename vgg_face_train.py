import sys
from data_access import ImageDaoKeras
from model_training import KerasTrain, get_exp_scheduler
from models.VGG_Face import VGG_Face


def train_vgg_face(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path=data_path)

    vgg_model = VGG_Face(dao_single_path).model
    lr_scheduler = get_exp_scheduler()

    trainer = KerasTrain(model=vgg_model,
                         name="VGGFace_1000ep_genki_modified_",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         batch_size=23,
                         epochs=1000,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def main(argv):
    assert len(argv) > 0
    data_path = argv[0]
    print("Training VGG Face on data from", data_path)
    train_vgg_face(data_path)


if __name__ == "__main__":
    main(sys.argv[1:])