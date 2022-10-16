from src import registry
from src.evaluation import metrics
from collections import defaultdict


@registry.register('default-scorer')
class DefaultScorer(object):
    def _select_metrics(self, _metrics):
        if _metrics == 'all':
            return metrics.METRIC_FNS, metrics.METRIC_FNS_SCORES

        METRICS_FNS = {
            key: fn 
            for key, fn 
            in metrics.METRIC_FNS.items()
            if key in _metrics
        }

        METRIC_FNS_SCORES = {
            key: fn 
            for key, fn 
            in metrics.METRIC_FNS_SCORES.items()
            if key in _metrics
        }

        return METRICS_FNS, METRIC_FNS_SCORES


    def score_by_scores(self,
        preds_scores,
        labels,
        _metrics='all'
    ):
        _, METRIC_FNS_SCORES = self._select_metrics(_metrics)

        results = {}
        for metric, fn in METRIC_FNS_SCORES.items():
            results[metric] = fn(y_true=labels, y_score=preds_scores)

        return results


    def score_by_preds(self,
        preds,
        labels,
        _metrics='all'
    ):
        METRIC_FNS, _ = self._select_metrics(_metrics)

        results = {}
        for metric, fn in METRIC_FNS.items():
            results[metric] = fn(y_true=labels, y_pred=preds)

        return results


    def score(self, 
        preds, 
        labels,
        preds_scores=None,
        _metrics='all'
    ):
        results = self.score_by_preds(
            preds=preds,
            labels=labels,
            _metrics=_metrics
        )

        if preds_scores is not None:
            results_scores = self.score_by_scores(
                preds_scores=preds_scores,
                labels=labels,
                _metrics=_metrics
            )
            results = {**results, **results_scores}

        return results


@registry.register('trecis-scorer')
class TrecisScorer:
    def __init__(self,
        low_level_encoder,
        high_level_encoder=None, 
        actionable_info_types=[]
    ):
        self.low_level_encoder = low_level_encoder
        self.high_level_encoder = high_level_encoder
        self.actionable_info_types = actionable_info_types
        self._init_maps()


    def _init_maps(self):
        self.low_level_idx2name = {
            idx: name
            for idx, name 
            in enumerate(self.low_level_encoder.classes_)
        }
        self.low_level_name2idx = {
            name: idx 
            for idx, name 
            in self.low_level_idx2name.items()
        }
        self.high_level_idx2name = {
            idx: name
            for idx, name
            in enumerate(self.high_level_encoder.classes_)
        }
        self.high_level_name2idx = {
            name: idx 
            for idx, name 
            in self.high_level_idx2name.items()
        }

    
    def _select_metrics(self, _metrics):
        if _metrics == 'all':
            return metrics.METRIC_FNS, metrics.METRIC_FNS_SCORES

        METRICS_FNS = {
            key: fn 
            for key, fn 
            in metrics.METRIC_FNS.items()
            if key in _metrics
        }

        METRIC_FNS_SCORES = {
            key: fn 
            for key, fn 
            in metrics.METRIC_FNS_SCORES.items()
            if key in _metrics
        }
        return METRICS_FNS, METRIC_FNS_SCORES


    def _filter_actionable(self,
        preds,
        labels,
        preds_scores=None
    ):
        actionable_indices = [
            self.low_level_name2idx[name]
            for name in self.actionable_info_types
        ]
        
        if preds is not None:
            preds = preds[:, actionable_indices]
        if labels is not None:
            labels = labels[:, actionable_indices]
        if preds_scores is not None:
            preds_scores = preds_scores[:, actionable_indices]
        return preds, preds_scores, labels


    def score_by_scores(self,
        preds_scores,
        labels,
        with_actionable=False,
        _metrics='all'
    ):
        _, METRIC_FNS_SCORES = self._select_metrics(_metrics)

        results = {}
        for metric, fn in METRIC_FNS_SCORES.items():
            results[metric] = fn(y_true=labels, y_score=preds_scores)

        if with_actionable:
            _, preds_scores, labels = self._filter_actionable(
                preds=None,
                preds_scores=preds_scores,
                labels=labels
            )
            for metric, fn in METRIC_FNS_SCORES.items():
                results[metric+'_actionable'] = fn(y_true=labels, y_score=preds_scores)
        return results


    def score_by_preds(self,
        preds,
        labels,
        with_actionable=False,
        _metrics='all'
    ):
        METRIC_FNS, _ = self._select_metrics(_metrics)

        results = {}
        for metric, fn in METRIC_FNS.items():
            results[metric] = fn(y_true=labels, y_pred=preds)

        if with_actionable:
            preds, _, labels = self._filter_actionable(
                preds=preds,
                preds_scores=None,
                labels=labels
            )
            for metric, fn in METRIC_FNS.items():
                results[metric+'_actionable'] = fn(y_true=labels, y_pred=preds)
        return results


    def score_by_info_types(self,
        preds,
        labels,
        low_level=True,
        _metrics='all'
    ):
        idx2name = self.low_level_idx2name if low_level else self.high_level_idx2name

        if _metrics == 'all':
            METRIC_FNS_BIN = metrics.METRIC_FNS_BINARY
        else:
            METRIC_FNS_BIN = {
                key: fn 
                for key, fn 
                in metrics.METRIC_FNS_BINARY.items()
                if key in _metrics
            }

        results_by_it = {}
        for idx, name in idx2name.items():
            preds_by_it = preds[:, idx]
            labels_by_it = labels[:, idx]
            for metric, fn in METRIC_FNS_BIN.items():
                results_by_it[f'{name}_{metric}'] = fn(y_true=labels_by_it, y_pred=preds_by_it)
        return results_by_it


    def score_by_events(self,
        preds,
        labels,
        events,
        preds_scores=None,
        with_actionable=False,
        _metrics='all'
    ):
        preds_by_event = defaultdict(list)
        preds_scores_by_event = defaultdict(list)
        labels_by_event = defaultdict(list)

        if preds_scores is None:
            without_scores = True
            preds_scores = [None] * len(preds)

        iterable = zip(preds, preds_scores, labels, events)
        for _preds, _preds_scores, _labels, _event in iterable:
            preds_by_event[_event].append(_preds)
            labels_by_event[_event].append(_labels)

            if without_scores:
                preds_scores_by_event[_event] = None
            else:
                preds_scores_by_event[_event] = _preds_scores

        results_by_event = {}
        for event in preds_by_event.keys():
            results_by_event[event] = self.score(
                preds=preds_by_event[event],
                labels=labels_by_event[event],
                preds_scores=preds_scores_by_event[event],
                with_actionable=with_actionable,
                _metrics=_metrics
            )
        return results_by_event


    def score(self, 
        preds, 
        labels,
        preds_scores=None,
        with_actionable=False,
        _metrics='all'
    ):
        results = self.score_by_preds(
            preds=preds,
            labels=labels,
            with_actionable=with_actionable,
            _metrics=_metrics
        )

        if preds_scores is not None:
            results_scores = self.score_by_scores(
                preds_scores=preds_scores,
                labels=labels,
                with_actionable=with_actionable,
                _metrics=_metrics
            )
            results = {**results, **results_scores}
        return results