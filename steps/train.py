from image_gata_generator import *
import keras._tf_keras.keras
from keras._tf_keras.keras.applications.resnet50 import ResNet50

def feature_extractor(inputs):
    feature_extractor = ResNet50(input_shape=(G.img_height, G.img_width, G.RGB),
                                               include_top=False,
                                               weights='imagenet')(inputs)
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
    inputs = keras.Input(shape=(G.img_height,G.img_width,G.RGB))

    classification_output = final_model(inputs)
    model = keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    keras.backend.clear_session()
    return model

if (__name__) == '__main__':
    base_dir = Path(os.getcwd()).parent
    model_path = os.path.join(base_dir, 'models/resnet50.keras')
    generator_train,generator_validation =  train_val_generators()
    model = get_resnet50()
    history = model.fit(generator_train,
                        validation_data=generator_validation,
                        epochs=G.nb_epochs,
                        callbacks = G.callbacks )
    keras.backend.clear_session()
    print(model.evaluate(generator_validation))
    model.save(model_path)
    
    #reconstructed_model = keras.models.load_model(model_path)
