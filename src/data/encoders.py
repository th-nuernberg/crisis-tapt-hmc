import json
import numpy as np
from src import registry
from sklearn.preprocessing import MultiLabelBinarizer


@registry.register('prio2multiclass')
class PrioIntegerEncoder:
    def __init__(self, file):
        self.encoder = json.load(open(file, 'r'))
        self.inverse_encoder = {
            value: key
            for key, value in self.encoder.items()
        }


    def transform(self, label):
        return self.encoder[label]


    def inverse_transform(self, label):
        return self.inverse_encoder[label]


@registry.register('prio2regression')
class PrioRegressionEncoder:
    def __init__(self, file):
        self.encoder = json.load(open(file, 'r'))
        self.inverse_encoder = {
            value: key
            for key, value in self.encoder.items()
        }


    def transform(self, label):
        return self.encoder[label]


    def inverse_transform(self, label):
        return self.inverse_encoder[label]


@registry.register('infotype2multilabel')
class InfoTypeEncoder:
    def __init__(self, file, _type='low_info_type'):
        info_types = json.load(open(file, 'r'))
        self.encoder = MultiLabelBinarizer().fit(
            [info_types[_type]]
        )


    def transform(self, label):
        label = np.array([label])
        encoding = self.encoder.transform(label)[0]
        return encoding.tolist()


    def inverse_transform(self, label):
        label = np.array([label])
        decoding = self.encoder.inverse_transform(label)[0]
        return decoding.tolist()