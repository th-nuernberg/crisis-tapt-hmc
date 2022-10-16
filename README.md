# Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning

This repository supplements our paper "Enhancing Crisis-Related Tweet Classification with Entity-Masked Language Modeling and Multi-Task Learning" accepted for the Workshop on 
NLP for Positive Impact @ EMNLP2022. This is a refactored version of the code used for the results in the paper which highlights the main components of the proposed approach.

## Materials

Due to the Twitter Developer Agreement and Policy we are not allowed to share the datasets in order to comply with the content redistribution.
The datasets and labels can be directly downloaded at [TREC Incident Streams](http://dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html).
However, we share the preprocessed datasets and extracted entities upon request (philipp.seeberger@th-nuernberg.de). The datasets are expected as json list in the following format:

```python
{
    "post_id": "...", 
    "text": "Nashville still cleaning up 2 months later after tornado https:...", 
    #"text": "[LOCATION] still cleaning up [NUMBER] months later after tornado [URL]", 
    "high_info_type": ["Other"], 
    "low_info_type": ["Irrelevant"], 
    "priority": "Low"
}
```

## Task-Adaptive Pretraining

Pretrain the BERT model with Masked-Language Modeling:

```bash
# Masked-Language Modeling (MLM)
python3 -m src.pretrain \
--config configs/tapt/pretrain_mlm.yaml
```

```bash
# Proposed Entity-Masked-Language Modeling (E-MLM)
python3 -m src.pretrain \
--config configs/tapt/pretrain_emlm.yaml
```

## Finetuning

Finetune the classification model with the pretrained BERT model:

```bash
# Single-Task
python3 -m src.finetune_st \
--config configs/st/<configuration-file>
```

```bash
# Multi-Task / Hierarchical
python3 -m src.finetune_mt \
--config configs/mt/<configuration-file> \
--model <model-type>
```

The model type must match the classification-head used in the configuration:

`finetune_emlm_lcl.yaml`: "lcl" <br/>
`finetune_emlm_lcpn.yaml`: "lcpn" <br/>
`finetune_emlm_hmcn_local.yaml`: "hmcn_local" <br/>
`finetune_emlm_hmcn_global.yaml`: "hmcn_global" <br/>

## Evaluation

Print the scoring metrics to the console:

```bash
# Single-Task
python3 -m src.eval \
--experiment <experiment-dir> \
--with-low-labels
```

```bash
# Multi-Task / Hierarchical
python3 -m src.eval \
--experiment <experiment-dir> \
--with-low-labels \
--with-high-labels
```