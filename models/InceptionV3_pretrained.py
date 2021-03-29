#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 00:43:34 2021

@author: rimashahbazyan
"""

from keras.applications import inception_v3
from keras.models       import Model
from keras.layers import Dense, experimental, Dropout
from keras import Input, Sequential

class InceptionV3_Pretrained:
    def __init__(self,
                 dataset=None):
        """
        Inception V3  model with pretrained weight from imagenet.

        :param dataset: ImageDaoKeras or ImageDaoCustom object
        """
        assert dataset is not None
        
        data_augmentation = Sequential(
            [
                experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(dataset.IMG_HEIGHT, dataset.IMG_WIDTH, 3)),
                experimental.preprocessing.RandomRotation(0.15),
                experimental.preprocessing.RandomZoom(0.1),
            ]
        )
        
        preprocess_input = inception_v3.preprocess_input
        
        original_model    = inception_v3()
        bottleneck_input  = original_model.get_layer(index=0).input
        bottleneck_output = original_model.get_layer(index=41).output
        bottleneck_model  = Model(inputs=bottleneck_input,           
                                  outputs=bottleneck_output)
        
        for layer in bottleneck_model.layers:
            layer.trainable = False
        
        prediction_layer = Dense(2, activation='softmax')
        
        inputs =  Input(shape=(dataset.IMG_HEIGHT, dataset.IMG_WIDTH, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = bottleneck_model(x, training=False)
        
        outputs = prediction_layer(x)
        

        self.model = Model(inputs, outputs)