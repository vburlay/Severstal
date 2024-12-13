import logging
import yaml
import mlflow
from steps.image_data_generator import *
from steps.train import get_resnet50,dice_coef
import os
import steps.preproc as pr
from steps.predict import Predictor
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
def main():
    with open ('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    mlflow.set_experiment("Model Training Experiments")

    with mlflow.start_run() as run:
        # Load data
        path = os.path.join(os.getcwd(), config['data']['train_path'])
        pr.extract_data(path)
        filename = os.path.join(os.getcwd(), config['data']["train_csv"])
        source_dir = os.path.join(os.getcwd(), config['data']["train_images"])
        pr.parse_data_from_input(filename, source_dir, path)
        logging.info("Data ingestion completed successfully")

        # Prepare and train model
        generator_train, generator_validation = train_val_generators()

        model = get_resnet50()
        model.fit(generator_train, validation_data=generator_validation,
                  epochs= config['train']['nb_epochs'],
                                           callbacks=G.callbacks)
        model_path = os.path.join(os.getcwd(), config['model']['store_path'])
        #model.save(model_path,overwrite=True)

        model = keras.models.load_model(model_path)

        mlflow.keras.log_model(model, "resnet50")
        logging.info("Model training completed successfully")

        # Evaluate model
        predictor = Predictor()
        accuracy, roc, precision, recall, f1, report= predictor.evaluate_model(
                                                             generator_train)
        logging.info("Model evaluation completed successfully")

        # Tags
        mlflow.set_tag("release.version", "2.0.0")
        mlflow.set_tag("preprocessing","Size-(224:224)")

        # Log metrics
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc", roc)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1', f1)

        # Register the model
        model_name = "classifications_model"
        model_uri = f"runs:/{run.info.run_id}/{config['model']['name']}"
        mlflow.register_model(model_uri, model_name)

        logging.info("MLflow tracking completed successfully")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {config['model']['name']}")
        print(
            f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc:.4f}")
        print(f"\n{report}")
        print("=====================================================\n")
def unet():
    with open ('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    mlflow.set_experiment("Model Training Experiments")

    with mlflow.start_run() as run:
        model_path = os.path.join(os.getcwd(), config['model']['store_path'])
        model = keras.models.load_model(model_path,custom_objects={
                                        'dice_coef':dice_coef})
        mlflow.keras.log_model(model, "unet")
        logging.info("Model training completed successfully")

        # Register the model
        model_name = "segmentation_model"
        model_uri = f"runs:/{run.info.run_id}/{config['model']['name']}"
        mlflow.register_model(model_uri, model_name)

        logging.info("MLflow tracking completed successfully")

if __name__ == "__main__":
    main()
    unet()



