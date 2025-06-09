# Steel Defect Detection

![image](/images/Steel.png ) 

>Defect Detection based on classification and localization.

## Table of Contents 

* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)



## General Information

> To improve the efficiency of steel production, this program will help engineers automate the process of locating and classifying surface defects on steel plate.
> This program uses the following algorithms:
 > - Classification (binary ResNet50-model).
 > - Segmentation (U-Net).

 > Dataset: "Severstal" comes from Kaggle [_here_](https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview).

## Technologies Used
- Python - version 3.10
- AWS S3
- MLflow

## Features
- Pandas,numpy,keras,tensorflow,scikit-learn,mlflow,segmentation_models
- ResNet50, Unet

## Screenshots
- Classes

![Example screenshot](/images/classes.png)

- Roc curve (Classification-ResNet50)

![Example screenshot](/images/roc.png)

- Dice coefficient

![Example screenshot](/images/Dice coef.png)

- Segmentation

![Example screenshot](/images/segmentation.png)

## Setup

It is necessary to install the following Python-Packages additionally: 
```r
segmentation_models, keras, tensorflow, pandas , matplotlib 
```

## Usage

* Preparation
```r
def find_files(path, filename):
    data_csv = pd.read_csv(filename)
    for root, dirs, files in os.walk(path):
        data_img = pd.DataFrame(files, columns=["ImageId"])
    dfs_dictionary = {"DF1": data_csv.ImageId, "DF2": data_img}
    df = pd.concat(dfs_dictionary)
    df = df.drop_duplicates(keep=FALSE)
    return df["ImageId"].tolist()

def extract_data(path):
    if not os.path.exists(path):
        os.mkdir(path)
        normal = os.path.join(path, "normal")
        os.mkdir(normal)
        defects = os.path.join(path, "defect")
        os.mkdir(defects)

def parse_data_from_input(filename, source_dir, target_dir):
    ls = find_files(source_dir, filename)
    for row in ls:
        temp_test_data = source_dir + "/" + row
        ziel_dir = os.path.join(target_dir, "normal/")
        final_val_data = ziel_dir + row
        shutil.copyfile(temp_test_data, final_val_data)
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        # Skip header
        next(csv_reader, None)
        for row in csv_reader:
            ziel_dir = os.path.join(target_dir, "defect/")
            temp_test_data = source_dir + "/" + row[0]
            final_val_data = ziel_dir + row[0]
            shutil.copyfile(temp_test_data, final_val_data)
```
* Makefile
```r
python = python-env/bin/python
pip = python-env/bin/pip
setup:
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt
run:
	$(python) main.py
mlflow:
	python-env/bin/mlflow ui
test:
	$(python) -m pytest
```                 
* MLflow
```r
 # Tags
        mlflow.set_tag("release.version", "2.0.0")
        mlflow.set_tag("preprocessing", "Size-(224:224)")

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
```

## Project Status

Project is complete 


## Room for Improvement

* By using ResNet101 model, an increase in the accuracy of the prediction could be achieved.


## Contact
Created by [Vladimir Burlay](wladimir.burlay@gmail.com) - feel free to contact me!

