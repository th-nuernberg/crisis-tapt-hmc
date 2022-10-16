from functools import partial
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)


METRIC_FNS = {
    'accuracy': accuracy_score,
    'f1_micro': partial(f1_score, average='micro'),
    'f1_macro': partial(f1_score,  average='macro'),
    'f1_weighted': partial(f1_score, average='weighted'),
    'precision_micro': partial(precision_score, average='micro'),
    'precision_macro': partial(precision_score, average='macro'),
    'precision_weighted': partial(precision_score, average='weighted'),
    'recall_micro': partial(recall_score, average='micro'),
    'recall_macro': partial(recall_score, average='macro'),
    'recall_weighted': partial(recall_score, average='weighted')
}


METRIC_FNS_BINARY = {
    'f1_binary': partial(f1_score, average='binary'),
    'precision_binary': partial(precision_score, average='binary'),
    'recall_binary': partial(recall_score, average='binary')
}


METRIC_FNS_SCORES = {    
    'avg_precision_micro': partial(average_precision_score, average='micro'),
    'avg_precision_macro': partial(average_precision_score, average='macro'),
    'avg_precision_weighted': partial(average_precision_score, average='weighted'),
    'roc_auc_micro': partial(roc_auc_score, average='micro'),
    'roc_auc_macro': partial(roc_auc_score, average='macro'),
    'roc_auc_weighted': partial(roc_auc_score, average='weighted')
}