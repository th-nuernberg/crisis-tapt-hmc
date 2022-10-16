import torch
from src import registry
from transformers import (
    DataCollatorForLanguageModeling
)


# masked language modeling
@registry.register('default-masking-collator')
def DataCollatorForDefaultMasking(tokenizer, **kwargs):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        **kwargs
    )


# entity-masked language modeling
@registry.register('entity-masking-collator')
class DataCollatorForEntityMasking:
    def __init__(self,
        tokenizer,
        entity_prob,
        non_entity_prob,
        entity_tokens,
        ignore_idx=-100
    ):
        self.tokenizer = tokenizer
        self.entity_prob = entity_prob
        self.non_entity_prob = non_entity_prob
        self.entity_tokens = entity_tokens
        self.ignore_idx = ignore_idx


    def __call__(self, batch):
        batch = self.tokenizer.pad(batch, return_tensors='pt')
        batch['input_ids'], batch['labels'] = self.mask_tokens(
            inputs=batch['input_ids']
        )
        return batch


    def mask_tokens(self, inputs):
        labels = inputs.clone()
        tokenizer = self.tokenizer

        # prepare token ids
        special_tokens = tokenizer.all_special_tokens
        special_tokens_idx = tokenizer.convert_tokens_to_ids(special_tokens)
        entity_tokens_idx = tokenizer.convert_tokens_to_ids(self.entity_tokens)
        special_tokens_idx = list(set(special_tokens_idx) - set(entity_tokens_idx))
        
        # create tokens mask
        special_tokens_mask = torch.tensor([
            [1 if token in special_tokens_idx else 0
             for token in tokens] 
            for tokens in labels.tolist()
        ], dtype=torch.bool)

        entity_tokens_mask = torch.tensor([
            [1 if token in entity_tokens_idx else 0
             for token in tokens]
            for tokens in labels.tolist()
        ], dtype=torch.bool)

        # create masking probs
        prob_matrix = torch.full(labels.shape, self.non_entity_prob)
        prob_matrix.masked_fill_(special_tokens_mask, value=0.)
        prob_matrix.masked_fill_(entity_tokens_mask, value=self.entity_prob)

        # ignore unconsidered tokens
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = self.ignore_idx

        # 80% tokens masked out
        prob_matrix_masked = torch.full(labels.shape, 0.8)
        replaced_indices = torch.bernoulli(prob_matrix_masked).bool()
        replaced_indices = replaced_indices & masked_indices
        inputs[replaced_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% tokens randomly replaced
        prob_matrix_random = torch.full(labels.shape, 0.5)
        random_indices = torch.bernoulli(prob_matrix_random).bool()
        random_indices = random_indices & masked_indices & ~replaced_indices
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[random_indices] = random_words[random_indices]

        # 10% tokens unchanged
        return inputs, labels