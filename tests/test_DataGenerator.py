import unittest
import pandas as pd
import os

from steps.image_data_generator import DataGenerator
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import  get_preprocessing

class DataGeneratorTestCase(unittest.TestCase):
    def test_init(self):
        path = os.path.join(os.getcwd(), 'data')
        test = pd.read_csv(path + '/sample_submission.csv')
        defect: list[str] = test.sample(16).index
        test_batches = DataGenerator(test[test.index.isin(defect)],
                                     subset='test')
        self.assertEqual(test_batches.subset, 'test')
        self.assertEqual(test_batches.batch_size, 16)

    def test_generator(self):
        path = os.path.join(os.getcwd(), 'data')
        test = pd.read_csv(path + '/sample_submission.csv')
        defect: list[str] = test.sample(16).index
        test_batches = DataGenerator(test[test.index.isin(defect)],
                                        subset='test',
                                        preprocess=get_preprocessing('resnet50'))
        self.assertEqual(test_batches.info.__class__, dict)


