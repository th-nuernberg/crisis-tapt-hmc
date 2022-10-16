from src import utils
from src import registry
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM
)


@registry.register('bert-tokenizer')
def get_bert_tokenizer(
    model_name_or_path, 
    add_special_tokens=[], 
    use_fast=False, 
    cache_dir='hf'
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        cache_dir=cache_dir
    )

    if len(add_special_tokens) > 0:
        utils.logger.info(f'tokenizer - add special tokens: {add_special_tokens}')

        tokenizer.add_special_tokens(
            {'additional_special_tokens': [
                token
                for token in add_special_tokens
            ]}
        )
    return tokenizer


@registry.register('bert-encoder')
def get_bert_model(
    model_name_or_path,
    tokenizer=None,
    cache_dir='hf',
    output_hidden_states=False,
    output_attentions=False
):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions
    )

    lm = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir
    )

    if tokenizer:
        lm.resize_token_embeddings(
            len(tokenizer)
        )
    return lm


@registry.register('bert-mlm-encoder')
def get_bert_mlm_model(
    model_name_or_path,
    tokenizer=None,
    cache_dir='hf'
):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir
    )

    lm = AutoModelForMaskedLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir
    )

    if tokenizer:
        lm.resize_token_embeddings(
            len(tokenizer)
        )
    return lm