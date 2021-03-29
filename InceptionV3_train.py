#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:04:19 2021

@author: rimashahbazyan
"""

import sys
from data_access import ImageDaoKeras
from model_training import KerasTrain, get_exp_scheduler
from models.InceptionV3_pretrained import InceptionV3_Pretrained

def train_InceptionV3(data_path):
    assert data_path is not None

    dao_single_path = ImageDaoKeras(data_path=data_path)

    model = InceptionV3_Pretrained(dao_single_path).model
    lr_scheduler = get_exp_scheduler(decay_step=1000)

    trainer = KerasTrain(model=model,
                         name="InceptionV3_pretrained_1000ep_genki_modified",
                         train_data=dao_single_path.train_dataset,
                         valid_data=dao_single_path.valid_dataset,
                         iters=23,
                         epochs=1000,
                         lr_scheduler=lr_scheduler)

    trainer.fit_model()


def main(argv):
    assert len(argv) > 0
    data_path = argv[0]
    print("Training VGG on data from", data_path)
    train_InceptionV3(data_path)


if __name__ == "__main__":
    main(sys.argv[1:])