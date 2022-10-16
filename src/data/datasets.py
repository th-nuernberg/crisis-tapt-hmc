import datasets
from src import registry
from torch.utils.data import Dataset
from datasets.arrow_dataset import Batch


@registry.register('trecis-single-task-dataset')
class TrecisSingleTaskDataset(Dataset):
    def __init__(self, 
        data_path, 
        tokenizer,
        encoder,
        max_length=128, 
        pad_to_max_length=True, 
        truncation=True,
        mlm=False,
        cache_dir='hf'
    ):
        padding = 'max_length' if pad_to_max_length else False
        self.label_key = encoder['key']
        self.label_encoder = registry.make(encoder)

        dataset = datasets.Dataset.from_json(
            path_or_paths=data_path,
            cache_dir=cache_dir
        )

        def process_fn(post):
            if isinstance(post, Batch):
                post[self.label_key] = [
                    self.label_encoder.transform(label)
                    for label in post[self.label_key]
                ]
            else:
                post[self.label_key] = self.label_encoder.transform(
                    post[self.label_key]
                )

            tokenized = tokenizer(
                post['text'],
                padding=padding, 
                max_length=max_length, 
                truncation=truncation
            )

            post.update(tokenized)
            return post

        dataset = dataset.map(
            process_fn,
            batched=True
        )

        if mlm:
            columns = [
                'input_ids',
                'token_type_ids',
                'attention_mask'
            ]
        else:
            columns = [
                'input_ids',
                'token_type_ids',
                'attention_mask',
                self.label_key
            ]

        dataset.set_format(
            type='torch',
            columns=columns
        )
        self.dataset = dataset


    def __len__(self):
        return self.dataset.__len__()


    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


@registry.register('trecis-multi-task-dataset')
class TrecisMultiTaskDataset(Dataset):
    def __init__(self, 
        data_path, 
        tokenizer,
        encoder_high_lvl,
        encoder_low_lvl,
        max_length=128, 
        pad_to_max_length=True, 
        truncation=True,
        mlm=False,
        cache_dir='hf'
    ):
        padding = 'max_length' if pad_to_max_length else False
        self.label_key_high_lvl = encoder_high_lvl['key']
        self.label_key_low_lvl = encoder_low_lvl['key']
        self.label_encoder_high_lvl = registry.make(encoder_high_lvl)
        self.label_encoder_low_lvl = registry.make(encoder_low_lvl)

        dataset = datasets.Dataset.from_json(
            path_or_paths=data_path,
            cache_dir=cache_dir
        )

        def process_fn(post):
            if isinstance(post, Batch):
                post[self.label_key_high_lvl] = [
                    self.label_encoder_high_lvl.transform(label)
                    for label in post[self.label_key_high_lvl]
                ]
                post[self.label_key_low_lvl] = [
                    self.label_encoder_low_lvl.transform(label)
                    for label in post[self.label_key_low_lvl]
                ]
            else:
                post[self.label_key_high_lvl] = self.label_encoder_high_lvl.transform(
                    post[self.label_key_high_lvl]
                )
                post[self.label_key_low_lvl] = self.label_encoder_low_lvl.transform(
                    post[self.label_key_low_lvl]
                )

            tokenized = tokenizer(
                post['text'],
                padding=padding, 
                max_length=max_length, 
                truncation=truncation
            )

            post.update(tokenized)
            return post

        dataset = dataset.map(
            process_fn,
            batched=True
        )

        if mlm:
            columns = [
                'input_ids',
                'token_type_ids',
                'attention_mask'
            ]
        else:
            columns = [
                'input_ids',
                'token_type_ids',
                'attention_mask',
                self.label_key_high_lvl,
                self.label_key_low_lvl
            ]

        dataset.set_format(
            type='torch',
            columns=columns
        )
        self.dataset = dataset


    def __len__(self):
        return self.dataset.__len__()


    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)