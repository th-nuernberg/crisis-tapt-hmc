import torch
import torch.nn as nn
from transformers import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SingleTaskTrainer(Trainer):
    def __init__(self, 
        loss_fn, 
        label_key,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.label_key = label_key


    def compute_loss(self, 
        model, 
        inputs, 
        return_outputs=False
    ):
        labels = inputs.pop(self.label_key)
        outputs = model(**inputs)
        if type(self.loss_fn) == nn.BCEWithLogitsLoss:
            loss = self.loss_fn(outputs, labels.float())
        else:
            loss = self.loss_fn(outputs, labels)

        return (loss, (loss, outputs)) if return_outputs else loss


class MultiTaskTrainer(Trainer):
    def __init__(self, 
        loss_fn_high_lvl,
        loss_fn_low_lvl, 
        label_key_high_lvl,
        label_key_low_lvl,
        loss_weight_high_lvl,
        loss_weight_low_lvl,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_fn_high_lvl = loss_fn_high_lvl
        self.loss_fn_low_lvl = loss_fn_low_lvl
        self.label_key_high_lvl = label_key_high_lvl
        self.label_key_low_lvl = label_key_low_lvl
        self.loss_weight_high_lvl = loss_weight_high_lvl
        self.loss_weight_low_lvl = loss_weight_low_lvl


    def compute_loss(self, 
        model, 
        inputs, 
        return_outputs=False
    ):
        labels_high_lvl = inputs.pop(self.label_key_high_lvl)
        labels_low_lvl = inputs.pop(self.label_key_low_lvl)

        outputs = model(**inputs)
        logits_high_lvl = outputs[0]
        logits_low_lvl = outputs[1]

        loss_high_lvl = self.loss_fn_high_lvl(
            logits_high_lvl, 
            labels_high_lvl.float()
        )

        loss_low_lvl = self.loss_fn_low_lvl(
            logits_low_lvl, 
            labels_low_lvl.float()
        )

        loss = (self.loss_weight_high_lvl * loss_high_lvl + 
                self.loss_weight_low_lvl * loss_low_lvl)

        return (loss, (loss, outputs)) if return_outputs else loss


class HierarchicalGlobalTaskTrainer(Trainer):
    def __init__(self, 
        loss_fn_high_lvl,
        loss_fn_low_lvl,
        loss_fn_global,
        loss_weight_high_lvl,
        loss_weight_low_lvl,
        loss_weight_global,
        label_key_high_lvl,
        label_key_low_lvl,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_fn_high_lvl = loss_fn_high_lvl
        self.loss_fn_low_lvl = loss_fn_low_lvl
        self.loss_fn_global = loss_fn_global
        self.label_key_high_lvl = label_key_high_lvl
        self.label_key_low_lvl = label_key_low_lvl
        self.loss_weight_high_lvl = loss_weight_high_lvl
        self.loss_weight_low_lvl = loss_weight_low_lvl
        self.loss_weight_global = loss_weight_global


    def compute_loss(self, 
        model, 
        inputs, 
        return_outputs=False
    ):
        labels_high_lvl = inputs.pop(self.label_key_high_lvl)
        labels_low_lvl = inputs.pop(self.label_key_low_lvl)
        labels_global = torch.cat([labels_high_lvl, labels_low_lvl], dim=-1)

        outputs = model(**inputs)
        logits_high_lvl = outputs[0]
        logits_low_lvl = outputs[1]
        logits_global = outputs[2]

        logits_global = 0.5 * torch.cat([logits_high_lvl, logits_low_lvl], axis=-1) + 0.5 * logits_global

        loss_high_lvl = self.loss_fn_high_lvl(
            logits_high_lvl, 
            labels_high_lvl.float()
        )

        loss_low_lvl = self.loss_fn_low_lvl(
            logits_low_lvl, 
            labels_low_lvl.float()
        )

        loss_global = self.loss_fn_global(
            logits_global,
            labels_global.float()
        )

        loss_local = (self.loss_weight_high_lvl * loss_high_lvl + 
                      self.loss_weight_low_lvl * loss_low_lvl)

        loss = loss_global + self.loss_weight_global * loss_local
                
        return (loss, (loss, outputs)) if return_outputs else loss


class HierarchicalParentNodeTaskTrainer(Trainer):
    def __init__(self, 
        loss_fn_high_lvl,
        loss_fn_low_lvl,
        loss_weight_high_lvl,
        loss_weight_low_lvl,
        label_key_high_lvl,
        label_key_low_lvl,
        hierarchy,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_fn_high_lvl = loss_fn_high_lvl
        self.loss_fn_low_lvl = loss_fn_low_lvl
        self.label_key_high_lvl = label_key_high_lvl
        self.label_key_low_lvl = label_key_low_lvl
        self.loss_weight_high_lvl = loss_weight_high_lvl
        self.loss_weight_low_lvl = loss_weight_low_lvl
        self.hierarchy = hierarchy


    def compute_loss(self, 
        model, 
        inputs, 
        return_outputs=False
    ):
        labels_high_lvl = inputs.pop(self.label_key_high_lvl)
        labels_low_lvl = inputs.pop(self.label_key_low_lvl)

        outputs = model(**inputs)
        logits_high_lvl = outputs[0]
        logits_low_lvl = outputs[1]

        loss_high_lvl = self.loss_fn_high_lvl(
            logits_high_lvl, 
            labels_high_lvl.float()
        )

        low_level_indices = []
        for idx in range(self.hierarchy.num_labels_high_lvl):
            low_level_indices.extend(self.hierarchy.get_low_level_indices(idx))
        logits_low_lvl = logits_low_lvl[:, low_level_indices]

        loss_low_lvl = self.loss_fn_low_lvl(
            logits_low_lvl, 
            labels_low_lvl.float()
        )

        loss = (self.loss_weight_high_lvl * loss_high_lvl + 
                self.loss_weight_low_lvl * loss_low_lvl)
                
        return (loss, (loss, outputs)) if return_outputs else loss