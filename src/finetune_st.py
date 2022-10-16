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
from src.models.trainers import SingleTaskTrainer

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

    model = registry.make(config['model'])
    loss_fn = registry.make(config['loss'])

    ###

    ### training setup

    scorer = registry.make({'name': 'default-scorer', 'params': {}})

    def compute_metrics(eval_preds):
        preds_logits, labels = eval_preds
        preds_probs = np.vectorize(utils.sigmoid)(preds_logits)
        preds = preds_probs.copy()
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        return scorer.score(preds=preds, labels=labels)

    label_key = config['train_dataset']['params']['encoder']['key']
    training_args.label_names = [label_key]
    trainer = SingleTaskTrainer(
        model=model,
        loss_fn=loss_fn, 
        label_key=label_key,
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
        preds_logits = results.predictions
        metrics = results.metrics
        metrics['eval_samples'] = len(eval_dataset)
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
        preds_probs = np.vectorize(utils.sigmoid)(preds_logits)
        preds = preds_probs.copy()
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        dataset = eval_dataset.dataset
        dataset = dataset.remove_columns(['input_ids', 'token_type_ids', 'attention_mask'])
        dataset = dataset.add_column(f'{label_key}_preds', preds.tolist())
        dataset = dataset.add_column(f'{label_key}_preds_probs', preds_probs.tolist())
        dataset.to_json(os.path.join(exp_dir, 'eval_predictions.json'))

    ###

    ### testing

    if training_args.do_predict:
        results = trainer.predict(test_dataset)
        preds_logits = results.predictions
        metrics = results.metrics
        metrics['test_samples'] = len(test_dataset)
        trainer.log_metrics('test', metrics)
        trainer.save_metrics('test', metrics)
        preds_probs = np.vectorize(utils.sigmoid)(preds_logits)
        preds = preds_probs.copy()
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        dataset = test_dataset.dataset
        dataset = dataset.remove_columns(['input_ids', 'token_type_ids', 'attention_mask'])
        dataset = dataset.add_column(f'{label_key}_preds', preds.tolist())
        dataset = dataset.add_column(f'{label_key}_preds_probs', preds_probs.tolist())
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
    main(parser.parse_args())