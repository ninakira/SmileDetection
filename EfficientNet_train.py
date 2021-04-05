import sys
import tensorflow as tf
from data_access import ImageDaoKerasBigData
from model_training import KerasTrain
from models.EfficientNet import EfficientNetV0
from config import get_train_valid_paths


def train_frozen(train_path, valid_path):

    assert train_path is not None
    assert valid_path is not None

    dao = ImageDaoKerasBigData(train_path=train_path, validation_path=valid_path)

    vgg_model = EfficientNetV0().model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    trainer = KerasTrain(model=vgg_model,
                         name="VGGFace_400ep_celeba_aug",
                         train_data=dao.train_dataset,
                         valid_data=dao.valid_dataset,
                         iters=5974,
                         epochs=400,
                         with_early_stop=True,
                         optimizer=optimizer)

    trainer.fit_model()

def main(argv):
    data_path = argv[0] if len(argv) > 0 else "augmented_celeba"
    print("Training VGG Face on data from", data_path)
    train_path, valid_path = get_train_valid_paths(data_path)
    train_frozen(train_path=train_path, valid_path=valid_path)


if __name__ == "__main__":
    main(sys.argv[1:])
