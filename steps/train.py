from steps.image_data_generator import *
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import keras._tf_keras.keras
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras import backend as K

from segmentation_models import Unet
from segmentation_models import get_preprocessing


def feature_extractor(inputs):
    feature_extractor = ResNet50(
        input_shape=(G.img_height, G.img_width, G.RGB),
        include_top=False,
        weights="imagenet",
    )(inputs)
    return feature_extractor


def classifier(inputs):
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(1, activation="sigmoid", name="classification")(x)
    return x


def final_model(inputs):
    resnet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output


def get_resnet50():
    inputs = keras.Input(shape=(G.img_height, G.img_width, G.RGB))

    classification_output = final_model(inputs)
    model = keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    keras.backend.clear_session()
    return model
def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

class get_unet():
    def __init__(self):
        self.BACKBONE = 'resnet50'
        self.base_dir = Path(os.getcwd())
        self.path = os.path.join(self.base_dir, "data")
        self.preprocess_input = get_preprocessing(self.BACKBONE)

    def dats(self):
        self.train = pd.read_csv(self.path + "/train.csv")
        train2 = self.train.pivot(index="ImageId", columns="ClassId",
                             values="EncodedPixels")
        train2.fillna("", inplace=True)
        train2["count"] = np.sum(train2.iloc[:] != "", axis=1).values
        train2 = pd.DataFrame(train2.to_records())
        train2.rename(columns={"1": "e1", "2": "e2", "3": "e3", "4": "e4"},
                      inplace=True)
        generator=DataGenerator(train2, shuffle=True,
                         preprocess=self.preprocess_input)
        return generator
    def model(self):
        model = Unet(
        self.BACKBONE ,
        encoder_weights="imagenet",
        input_shape=(128, 800, 3),
        classes=4,
        activation="sigmoid")
        model.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=[dice_coef])

        return model

if (__name__) == "__main__":
    base_dir = Path(os.getcwd()).parent
    model_path = os.path.join(base_dir, "models/resnet50.keras")
    generator_train, _ = train_val_generators()
    model = get_resnet50()
    history = model.fit(generator_train, epochs=G.nb_epochs, callbacks=G.callbacks)
    keras.backend.clear_session()
    model.save(model_path)

    # reconstructed_model = keras.models.load_model(model_path)

    # U-Net
    unet = get_unet()
    u_model = unet.model()
    generator = unet.dats()
    # TRAIN MODEL
    u_model.fit(generator, epochs=2, verbose=2)
    model_path_unet = os.path.join(base_dir, "models/unet.keras")
    u_model.save(model_path_unet, overwrite=True)



