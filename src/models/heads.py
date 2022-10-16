import torch
import torch.nn as nn
from src import registry
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


@registry.register('single-task-head')
class SingleTaskModel(PreTrainedModel):
    def __init__(self, 
        encoder, 
        num_labels,
        dropout=0.1
    ):
        super().__init__(PretrainedConfig())

        self.lm = registry.make(encoder)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels
        ) 


    def forward(self, 
        input_ids, 
        attention_mask, 
        token_type_ids
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs['pooler_output'])
        logits = self.head(pooled)
        return logits


@registry.register('multi-task-head')
class MultiTaskModel(PreTrainedModel):
    def __init__(self, 
        encoder, 
        num_labels_high_lvl,
        num_labels_low_lvl,
        dropout=0.1
    ):
        super().__init__(PretrainedConfig())
        
        self.lm = registry.make(encoder)
        self.dropout = nn.Dropout(dropout)

        self.head_high_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_high_lvl
        )

        self.head_low_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_low_lvl
        )


    def forward(self, 
        input_ids, 
        attention_mask, 
        token_type_ids
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs['pooler_output'])
        logits_high_lvl = self.head_high_lvl(pooled)
        logits_low_lvl = self.head_low_lvl(pooled)
        return logits_high_lvl, logits_low_lvl


@registry.register('hierarchical-task-head')
class HierarchicalTaskModel(PreTrainedModel):
    def __init__(self, 
        encoder, 
        num_labels_high_lvl,
        num_labels_low_lvl,
        dropout=0.1
    ):
        super().__init__(PretrainedConfig())

        self.lm = registry.make(encoder)
        self.dropout = nn.Dropout(dropout)

        self.pooler_low_lvl = nn.Sequential(
            nn.Linear(
                self.lm.config.hidden_size,
                self.lm.config.hidden_size
            ),
            nn.Tanh()
        )

        self.head_high_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_high_lvl
        )

        self.head_low_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_low_lvl
        )


    def forward(self, 
        input_ids, 
        attention_mask, 
        token_type_ids
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        first_token_tensor = outputs['last_hidden_state'][:, 0]

        pooled_high_lvl = self.dropout(outputs['pooler_output'])
        pooled_low_lvl = self.dropout(self.pooler_low_lvl(pooled_high_lvl + first_token_tensor))

        logits_high_lvl = self.head_high_lvl(pooled_high_lvl)
        logits_low_lvl = self.head_low_lvl(pooled_low_lvl)
        return logits_high_lvl, logits_low_lvl


@registry.register('hierarchical-global-task-head')
class HierarchicalGlobalTaskModel(PreTrainedModel):
    def __init__(self, 
        encoder, 
        num_labels_high_lvl,
        num_labels_low_lvl,
        dropout=0.1
    ):
        super().__init__(PretrainedConfig())

        self.lm = registry.make(encoder)
        self.dropout = nn.Dropout(dropout)

        self.pooler_low_lvl = nn.Sequential(
            nn.Linear(
                self.lm.config.hidden_size,
                self.lm.config.hidden_size
            ),
            nn.Tanh()
        )

        self.pooler_global = nn.Sequential(
            nn.Linear(
                self.lm.config.hidden_size,
                self.lm.config.hidden_size
            ),
            nn.Tanh()
        )

        self.head_high_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_high_lvl
        )

        self.head_low_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            num_labels_low_lvl
        ) 

        self.head_global = nn.Linear(
            self.lm.config.hidden_size,
            num_labels_high_lvl + num_labels_low_lvl
        )


    def forward(self, 
        input_ids, 
        attention_mask, 
        token_type_ids
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        first_token_tensor = outputs['last_hidden_state'][:, 0]

        pooled_high_lvl = self.dropout(outputs['pooler_output'])
        pooled_low_lvl = self.dropout(self.pooler_low_lvl(pooled_high_lvl + first_token_tensor))
        pooled_global = self.dropout(self.pooler_global(pooled_low_lvl + first_token_tensor))

        logits_high_lvl = self.head_high_lvl(pooled_high_lvl)
        logits_low_lvl = self.head_low_lvl(pooled_low_lvl)
        logits_global = self.head_global(pooled_global)
        return logits_high_lvl, logits_low_lvl, logits_global


@registry.register('hierarchical-parent-node-task-head')
class HierarchicalParentNodeTaskModel(PreTrainedModel):
    def __init__(self, 
        encoder, 
        hierarchy,
        dropout=0.1
    ):
        super().__init__(PretrainedConfig())

        self.lm = registry.make(encoder)
        self.dropout = nn.Dropout(dropout)

        self.pooler_low_lvl = nn.Sequential(
            nn.Linear(
                self.lm.config.hidden_size,
                self.lm.config.hidden_size
            ),
            nn.Tanh()
        )

        self.head_high_lvl = nn.Linear(
            self.lm.config.hidden_size, 
            hierarchy.num_labels_high_lvl
        )

        self.heads_low_lvl = nn.ModuleList([
            nn.Linear(
                self.lm.config.hidden_size, 
                len(hierarchy.get_low_level_indices(idx))
            )
            for idx in range(hierarchy.num_labels_high_lvl)
        ])


    def forward(self, 
        input_ids, 
        attention_mask, 
        token_type_ids
    ):
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = self.dropout(outputs['pooler_output'])
        logits_high_lvl = self.head_high_lvl(pooled)

        logits_low_lvl = torch.cat([
            head_low_lvl(pooled)
            for head_low_lvl in self.heads_low_lvl
        ], dim=-1)
        return logits_high_lvl, logits_low_lvl