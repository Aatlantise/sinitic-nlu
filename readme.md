# Sinitic NLU & Transfer learning

We investigate transfer learning from Mandarin to other Sinitic languages (Cantonese, Wu, Hokkien).

## Download model first

This repository requires a local copy of the BERT model and Wikipedia dataset to run.

To download the resources, simply run

```angular2html
python download.py --lang=yue
```
where lang can be `yue` or `wuu`.

## Model pre-training

To continually pre-train on Mandarin BERT, simply run

```angular2html
python run.py --pretrain --lang=yue
```
where lang can be `yue` or `wuu`. 
Additional flags are available--see `run.py`.

## Fine-tuning

To fine-tune, the code requires the Cantonese UD file.
Download the [CoNLL-U file](https://github.com/UniversalDependencies/UD_Cantonese-HK/blob/dev/yue_hk-ud-test.conllu)
and place it in `data/`, then use the `conllu_2_pos_dataset()` function in `utils.py`.
