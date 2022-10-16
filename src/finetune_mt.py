import os
import shutil
import argparse
import numpy as np
from transformers import (
    TrainingArguments, 
    default_data_collator, 
    set_seed
)

from src import utils
from src import data
from src import models
from src import evaluation
from src import registry
from src.utils.hierarchy import TrecisHierarchy
from src.models.trainers import (
    MultiTaskTrainer, 
    HierarchicalGlobalTaskTrainer, 
    HierarchicalParentNodeTaskTrainer
)

from transformers.utils import logging
logging.set_verbosity_info()


def apply_seed(config, seed):
    exp_dir = config['experiment']['dir']
    exp_dir = os.path.join(exp_dir, 'seed'+str(seed))
    config['experiment']['dir'] = exp_dir

    train_path = config['train_dataset']['params']['data_path'].format(seed)
    eval_path = config['eval_dataset']['params']['data_path'].format(seed)
    test_path = config['test_dataset']['params']['data_path'].format(seed)

    config['train_dataset']['params']['data_path'] = train_path
    config['eval_dataset']['params']['data_path'] = eval_path
    config['test_dataset']['params']['data_path'] = test_path

    enc_config = config['model']['params']['encoder']
    model_path = enc_config['params']['model_name_or_path']

    if 'seed' in model_path:
        model_path = model_path.format(seed)
        config['model']['params']['encoder'] \
              ['params']['model_name_or_path'] = model_path


def main(args):
    config = utils.load_config(args.config)
    seed = config['experiment']['seed']
    apply_seed(config, seed)
    utils.logger.info(config)

    ### environment setup

    exp_config = config['experiment']
    exp_dir = os.path.join(exp_config['dir'], exp_config['name'])

    if exp_config['override']:
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, 'models'),
        logging_dir=os.path.join(exp_dir, 'logs'),
        seed=exp_config['seed'],
        optim='adamw_torch',
        **config['training']
    )

    ###

    ### init datasets, tokenizer and model

    np.random.seed(exp_config['seed'])
    set_seed(exp_config['seed'])

    tokenizer = registry.make(config['tokenizer'])
    train_dataset = registry.make(config['train_dataset'], tokenizer=tokenizer)
    eval_dataset = registry.make(config['eval_dataset'], tokenizer=tokenizer)
    test_dataset = registry.make(config['test_dataset'], tokenizer=tokenizer)

    config['model']['params']['encoder'] \
          ['params']['tokenizer'] = tokenizer

    if args.model == 'lcpn':
        hierarchy = TrecisHierarchy(
            file='materials/meta/infotypes.json',
            high_level_encoder=train_dataset.label_encoder_high_lvl,
            low_level_encoder=train_dataset.label_encoder_low_lvl
        )
        config['model']['params']['hierarchy'] = hierarchy

    model = registry.make(config['model'])
    loss_fn = registry.make(config['loss'])

    ###

    ### training setup

    scorer = registry.make({'name': 'default-scorer', 'params': {}})

    def compute_metrics(eval_preds):
        preds_logits, labels = eval_preds
        preds_logits_high, labels_high = preds_logits[0], labels[0]
        preds_logits_low, labels_low = preds_logits[1], labels[1]
        preds_probs_high = np.vectorize(utils.sigmoid)(preds_logits_high)
        preds_high = preds_probs_high.copy()
        preds_high[preds_high >= 0.5] = 1
        preds_high[preds_high < 0.5] = 0

        if args.model == 'lcpn':
            low_level_indices = []
            for idx in range(hierarchy.num_labels_high_lvl):
                low_level_indices.extend(hierarchy.get_low_level_indices(idx))
            preds_logits_low = preds_logits_low[:, low_level_indices]

        preds_probs_low = np.vectorize(utils.sigmoid)(preds_logits_low)
        preds_low = preds_probs_low.copy()
        preds_low[preds_low >= 0.5] = 1
        preds_low[preds_low < 0.5] = 0
        scores_high = scorer.score(preds=preds_high, labels=labels_high)
        scores_low = scorer.score(preds=preds_low, labels=labels_low)
        scores_high = {f'high_{metric}': value for metric, value in scores_high.items()}
        return {**scores_low, **scores_high}

    label_key_high_lvl = config['train_dataset']['params']['encoder_high_lvl']['key']
    label_key_low_lvl = config['train_dataset']['params']['encoder_low_lvl']['key']
    training_args.label_names = [label_key_high_lvl, label_key_low_lvl]

    if args.model == 'lcl' or args.model == 'hmcn_local':
        trainer = MultiTaskTrainer(
            model=model,
            loss_fn_high_lvl=loss_fn['high_lvl'],
            loss_fn_low_lvl=loss_fn['low_lvl'],
            loss_weight_high_lvl=config['loss']['weight_high_lvl'],
            loss_weight_low_lvl=config['loss']['weight_low_lvl'],
            label_key_high_lvl=label_key_high_lvl,
            label_key_low_lvl=label_key_low_lvl,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics
        )
    elif args.model == 'hmcn_global':
        trainer = HierarchicalGlobalTaskTrainer(
            model=model,
            loss_fn_high_lvl=loss_fn['high_lvl'],
            loss_fn_low_lvl=loss_fn['low_lvl'],
            loss_fn_global=loss_fn['global'],
            loss_weight_high_lvl=config['loss']['weight_high_lvl'],
            loss_weight_low_lvl=config['loss']['weight_low_lvl'],
            loss_weight_global=config['loss']['weight_global'],
            label_key_high_lvl=label_key_high_lvl,
            label_key_low_lvl=label_key_low_lvl, 
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics
        )
    elif args.model == 'lcpn':
        trainer = HierarchicalParentNodeTaskTrainer(
            model=model,
            loss_fn_high_lvl=loss_fn['high_lvl'],
            loss_fn_low_lvl=loss_fn['low_lvl'],
            loss_weight_high_lvl=config['loss']['weight_high_lvl'],
            loss_weight_low_lvl=config['loss']['weight_low_lvl'],
            label_key_high_lvl=label_key_high_lvl,
            label_key_low_lvl=label_key_low_lvl, 
            hierarchy=hierarchy,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics
        )

    ###

    ### training

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(
                training_args.output_dir
            )

        metrics = train_result.metrics
        metrics['train_samples'] =  len(train_dataset)
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

    ###

    ### evaluation

    if training_args.do_eval:
        results = trainer.predict(eval_dataset)
        metrics = results.metrics
        preds_logits = results.predictions
        preds_logits_high = preds_logits[0]
        preds_logits_low = preds_logits[1]
        metrics['eval_samples'] = len(eval_dataset)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

        preds_probs_high = np.vectorize(utils.sigmoid)(preds_logits_high)
        preds_high = preds_probs_high.copy()
        preds_high[preds_high >= 0.5] = 1
        preds_high[preds_high < 0.5] = 0

        if args.model == 'lcpn':
            low_level_indices = []
            for idx in range(hierarchy.num_labels_high_lvl):
                low_level_indices.extend(hierarchy.get_low_level_indices(idx))
            preds_logits_low = preds_logits_low[:, low_level_indices]

        preds_probs_low = np.vectorize(utils.sigmoid)(preds_logits_low)
        preds_low = preds_probs_low.copy()
        preds_low[preds_low >= 0.5] = 1
        preds_low[preds_low < 0.5] = 0
        dataset = eval_dataset.dataset
        dataset = dataset.remove_columns(['input_ids', 'token_type_ids', 'attention_mask'])
        dataset = dataset.add_column(f'{label_key_high_lvl}_preds', preds_high.tolist())
        dataset = dataset.add_column(f'{label_key_high_lvl}_preds_probs', preds_probs_high.tolist())
        dataset = dataset.add_column(f'{label_key_low_lvl}_preds', preds_low.tolist())
        dataset = dataset.add_column(f'{label_key_low_lvl}_preds_probs', preds_probs_low.tolist())
        dataset.to_json(os.path.join(exp_dir, 'eval_predictions.json'))

    ###

    ### testing

    if training_args.do_predict:
        results = trainer.predict(test_dataset)
        metrics = results.metrics
        preds_logits = results.predictions
        preds_logits_high = preds_logits[0]
        preds_logits_low = preds_logits[1]
        metrics['test_samples'] = len(test_dataset)
        trainer.log_metrics('test', metrics)
        trainer.save_metrics('test', metrics)
        preds_probs_high = np.vectorize(utils.sigmoid)(preds_logits_high)
        preds_high = preds_probs_high.copy()
        preds_high[preds_high >= 0.5] = 1
        preds_high[preds_high < 0.5] = 0

        if args.model == 'lcpn':
            low_level_indices = []
            for idx in range(hierarchy.num_labels_high_lvl):
                low_level_indices.extend(hierarchy.get_low_level_indices(idx))
            preds_logits_low = preds_logits_low[:, low_level_indices]
            
        preds_probs_low = np.vectorize(utils.sigmoid)(preds_logits_low)
        preds_low = preds_probs_low.copy()
        preds_low[preds_low >= 0.5] = 1
        preds_low[preds_low < 0.5] = 0
        dataset = test_dataset.dataset
        dataset = dataset.remove_columns(['input_ids', 'token_type_ids', 'attention_mask'])
        dataset = dataset.add_column(f'{label_key_high_lvl}_preds', preds_high.tolist())
        dataset = dataset.add_column(f'{label_key_high_lvl}_preds_probs', preds_probs_high.tolist())
        dataset = dataset.add_column(f'{label_key_low_lvl}_preds', preds_low.tolist())
        dataset = dataset.add_column(f'{label_key_low_lvl}_preds_probs', preds_probs_low.tolist())
        dataset.to_json(os.path.join(exp_dir, 'test_predictions.json'))

    ###

    config_file = os.path.join(exp_dir, 'exp_config.yaml')
    utils.save_config(config_file, config)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['lcl', 'lcpn', 'hmcn_local', 'hmcn_global'],
        required=True
    )
    main(parser.parse_args())