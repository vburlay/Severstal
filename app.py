import mlflow
from steps.image_data_generator import *

logged_model = 'runs:/9be3261b65cf49fdb1812a0c0a3828c8/resnet50'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
generator_train, generator_validation = train_val_generators()
image_batch, y_true = next(iter(generator_validation))
y_pred = loaded_model.predict(image_batch)
predict = np.array([1 if x >= 0.5 else 0 for x in y_pred])
print(predict)