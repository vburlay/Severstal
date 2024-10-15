from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
import keras._tf_keras.keras
from dataclasses import dataclass
import os
from pathlib import Path
base_dir = Path(os.getcwd()).parent
train_dir = os.path.join(base_dir, 'data/ml_classification/')

@dataclass
class G:
    img_height = 224
    img_width = 224
    nb_epochs = 100
    batch_size = 64
    RGB = 3
    callbacks = [keras.callbacks.ReduceLROnPlateau(
                 monitor='val_loss', patience=5, mode='min', factor=0.2,
                 min_lr=1e-7, verbose=1)]

def train_val_generators():
        gen_train = ImageDataGenerator(
            rescale=1. / 255.,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2,
            preprocessing_function=preprocess_input)
        generator_train = gen_train.flow_from_directory(
            directory=train_dir,
            target_size=(G.img_height, G.img_width),
            batch_size=G.batch_size,
            shuffle=True,
            color_mode="rgb",
            class_mode="binary",
            subset='training',
            seed=21
        )
        generator_validation = gen_train.flow_from_directory(
            directory=train_dir,
            target_size=(G.img_height, G.img_width),
            batch_size=G.batch_size,
            shuffle=True,
            color_mode="rgb",
            class_mode="binary",
            subset='validation',
            seed=21
        )
        return generator_train, generator_validation