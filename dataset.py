import steps.preproc as pr
import os

if __name__ == '__main__':
    # ml-directory
    path = os.path.join(os.getcwd(), "data/ml_classification")
    pr.extract_data(path)
    # parse-data
    filename = os.path.join(os.getcwd(), "data/train.csv")
    source_dir = os.path.join(os.getcwd(), "data/train_images")
    pr.parse_data_from_input(filename, source_dir, path)
