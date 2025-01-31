import mlflow
import uvicorn

from steps.image_data_generator import *
from fastapi import FastAPI

app = FastAPI()
@app.get("/")
async def root():
    return {"detail_check": "OK", "model_version": 1}


@app.get("/predict")
async def predict():
    logged_model = 'runs:/9be3261b65cf49fdb1812a0c0a3828c8/resnet50'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    generator_train, generator_validation = train_val_generators()
    image_batch, y_true = next(iter(generator_validation))
    y_pred = loaded_model.predict(image_batch)
    predict = np.array([1 if x >= 0.5 else 0 for x in y_pred])

    return {"predicted_class": predict.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.0', port=8000)
