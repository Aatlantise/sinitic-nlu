# Sinitic NLU & Transfer learning

We investigate transfer learning from Mandarin to other Sinitic languages (Cantonese, Wu, Hokkien).

## Model pre-training

To continually pre-train on Mandarin BERT, simply run

```angular2html
python run.py --lang=yue
```
where lang can be `yue` or `wuu`.
