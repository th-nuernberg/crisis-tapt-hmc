from src import registry
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss
)


@registry.register('single-task-loss')
def create_st_loss(loss_name):
    loss_fn = None
    if loss_name == 'cross-entropy':
        loss_fn = CrossEntropyLoss()
    if loss_name == 'binary-cross-entropy':
        loss_fn = BCEWithLogitsLoss()
    if loss_name == 'mean-squared-error':
        loss_fn = MSELoss()
    return loss_fn


@registry.register('multi-task-loss')
def create_mt_loss(losses):
    loss_fns = {}
    for key, value in losses.items():
        if value == 'cross-entropy':
            loss_fns[key] = CrossEntropyLoss()
        if value == 'binary-cross-entropy':
            loss_fns[key] = BCEWithLogitsLoss()
        if value == 'mean-squared-error':
            loss_fns[key] = MSELoss()
    return loss_fns