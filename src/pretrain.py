import os
import shutil
import math
import argparse
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainingArguments, Trainer, set_seed

from src import utils
from src import data
from src import models
from src import registry


def apply_seed(config, seed):
    exp_dir = config['experiment']['dir']
    exp_dir = os.path.join(exp_dir, 'seed'+str(seed))
    config['experiment']['dir'] = exp_dir

    train_path = config['train_dataset']['params']['data_path'].format(seed)
    eval_path = config['eval_dataset']['params']['data_path'].format(seed)

    config['train_dataset']['params']['data_path'] = train_path
    config['eval_dataset']['params']['data_path'] = eval_path


def main(args):
    config = utils.load_config(args.config)
    seed = config['experiment']['seed']
    apply_seed(config, seed)
    utils.logger.info(config)

    ### environment setup

    exp_config = config['experiment']
    experiment_dir = os.path.join(exp_config['dir'], exp_config['name'])

    if exp_config['override']:
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(experiment_dir, 'models'),
        logging_dir=os.path.join(experiment_dir, 'logs'),
        seed=exp_config['seed'],
        optim='adamw_torch',
        **config['training']
    )

    ###

    ### check whether to continue from checkpoint

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir) and 
        training_args.do_train and not 
        training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            pass
            utils.logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    ###

    ### init datasets, tokenizer and model

    set_seed(exp_config['seed'])

    tokenizer = registry.make(config=config['tokenizer'])
    model = registry.make(config=config['model'], tokenizer=tokenizer)
    train_dataset = registry.make(config=config['train_dataset'], tokenizer=tokenizer)
    eval_dataset = registry.make(config=config['eval_dataset'], tokenizer=tokenizer)
    collator = registry.make(config=config['collator'], tokenizer=tokenizer)

    ### pretraining setup

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return {'accuracy': accuracy_score(
            y_pred=preds,
            y_true=labels
        )}

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits
    )

    ###

    ### training

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
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
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(eval_dataset)

        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')

        metrics['perplexity'] = perplexity
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    ###

    config_file = os.path.join(experiment_dir, 'exp_config.yaml')
    utils.save_config(config_file, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        required=True
    )
    main(parser.parse_args())