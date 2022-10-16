import json


class TrecisHierarchy:
    def __init__(self,
        file,
        high_level_encoder,
        low_level_encoder
    ):
        self.ontology = json.load(open(file, 'r'))['ontology']
        self.high_level_encoder = high_level_encoder
        self.low_level_encoder = low_level_encoder
        self.num_labels_high_lvl = len(high_level_encoder.encoder.classes_)
        self.num_labels_low_lvl = len(low_level_encoder.encoder.classes_)


    def get_low_level_indices(self, high_level_idx):
        high_level_class = self.high_level_encoder.encoder.classes_[high_level_idx]
        low_level_classes = self.ontology[high_level_class]
        low_level_encodings = self.low_level_encoder.transform(low_level_classes)
        low_level_indices = [idx for idx, label in enumerate(low_level_encodings) if label==1]
        return low_level_indices