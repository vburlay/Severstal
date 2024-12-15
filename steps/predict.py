import keras._tf_keras.keras
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self):
        self.model_path = os.path.join(os.getcwd(),
                                       'models/resnet50.keras')
        self.reconstructed_model = keras.models.load_model(self.model_path)

    def roc_cur(fpr_keras, tpr_keras, auc_keras):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras,
                 label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    def evaluate_model(self, generator_train):
        image_batch, y_true = next(iter(generator_train))
        y_pred = self.reconstructed_model.predict(image_batch)
        predict = np.array([1 if x >= 0.5 else 0 for x in y_pred])

        report = classification_report(y_true, predict, output_dict=True)

        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        fpr_keras, tpr_keras, _ = roc_curve(y_true, predict)
        roc = auc(fpr_keras, tpr_keras)
        return accuracy, roc, precision, recall, f1, report
