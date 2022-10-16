import os
import json
import argparse
import warnings
import numpy as np

from src import data
from src import evaluation
from src import registry

warnings.filterwarnings('ignore')


def main(args):
    eval_preds_file = os.path.join(args.experiment, args.eval_preds_file)
    test_preds_file = os.path.join(args.experiment, args.test_preds_file)

    actionable = [
        'MovePeople',
        'EmergingThreats',
        'NewSubEvent',
        'ServiceAvailable',
        'GoodsServices',
        'SearchAndRescue'
    ]

    low_level_encoder = registry.make({   
        'name': 'infotype2multilabel',
        'params': {
            'file': 'materials/meta/infotypes.json', 
            '_type': 'low_info_type'
        }
    })

    high_level_encoder = registry.make({   
        'name': 'infotype2multilabel',
        'params': {
            'file': 'materials/meta/infotypes.json', 
            '_type': 'high_info_type'
        }
    })

    scorer = registry.make({
        'name': 'trecis-scorer',
        'params': {
            'low_level_encoder': low_level_encoder.encoder,
            'high_level_encoder': high_level_encoder.encoder,
            'actionable_info_types': actionable
        }
    })


    if args.with_high_labels:
        with open(eval_preds_file, 'r') as f:
            eval_preds = []
            eval_labels = []
            for item in f:
                item = json.loads(item)
                eval_preds.append(item[args.high_preds_key])
                eval_labels.append(item[args.high_labels_key])
            eval_preds = np.array(eval_preds)
            eval_labels = np.array(eval_labels)

        with open(test_preds_file, 'r') as f:
            test_preds = []
            test_labels = []
            for item in f:
                item = json.loads(item)
                test_preds.append(item[args.high_preds_key])
                test_labels.append(item[args.high_labels_key])
            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)

        eval_metrics = scorer.score(
            preds=eval_preds,
            preds_scores=None,
            labels=eval_labels,
            with_actionable=False
        )

        test_metrics = scorer.score(
            preds=test_preds,
            preds_scores=None,
            labels=test_labels,
            with_actionable=False
        )

        print("### high level labels (development) ###")
        print(json.dumps(eval_metrics, indent=2))
        print("\n### high level labels (test) ###")
        print(json.dumps(test_metrics, indent=2))


    if args.with_low_labels:
        with open(eval_preds_file, 'r') as f:
            eval_preds = []
            eval_labels = []
            for item in f:
                item = json.loads(item)
                eval_preds.append(item[args.low_preds_key])
                eval_labels.append(item[args.low_labels_key])
            eval_preds = np.array(eval_preds)
            eval_labels = np.array(eval_labels)

        with open(test_preds_file, 'r') as f:
            test_preds = []
            test_labels = []
            for item in f:
                item = json.loads(item)
                test_preds.append(item[args.low_preds_key])
                test_labels.append(item[args.low_labels_key])
            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)

        eval_metrics = scorer.score(
            preds=eval_preds,
            preds_scores=None,
            labels=eval_labels,
            with_actionable=True
        )

        test_metrics = scorer.score(
            preds=test_preds,
            preds_scores=None,
            labels=test_labels,
            with_actionable=True
        )

        print("### Low level labels (development) ###")
        print(json.dumps(eval_metrics, indent=2))
        print("\n### Low level labels (test) ###")
        print(json.dumps(test_metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True
    )
    parser.add_argument(
        '--low-labels-key', 
        dest='low_labels_key', 
        type=str, 
        default='low_info_type'
    )
    parser.add_argument(
        '--high-labels-key', 
        dest='high_labels_key', 
        type=str, 
        default='high_info_type'
    )
    parser.add_argument(
        '--low-preds-key', 
        dest='low_preds_key', 
        type=str, 
        default='low_info_type_preds'
    )
    parser.add_argument(
        '--high-preds-key', 
        dest='high_preds_key', 
        type=str, 
        default='high_info_type_preds'
    )
    parser.add_argument(
        '--with-low-labels',
        dest='with_low_labels',
        action='store_true'
    )
    parser.add_argument(
        '--with-high-labels',
        dest='with_high_labels',
        action='store_true'
    )
    parser.add_argument(
        '--eval-preds-file',
        dest='eval_preds_file',
        type=str,
        default='eval_predictions.json'
    )
    parser.add_argument(
        '--test-preds-file',
        dest='test_preds_file',
        type=str,
        default='test_predictions.json'
    )
    main(parser.parse_args()) 