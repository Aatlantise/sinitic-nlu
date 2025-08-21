from datasets import load_from_disk, Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertPreTrainedModel,
    BertModel,
    BertConfig,
)
from transformers.modeling_outputs import TokenClassifierOutput
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from utils import get_subset
import torch.nn as nn


class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/bert-base-chinese-local", scratch=False):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.model_dir = model_dir
        self.tokenized_ds = None
        self.lm_dataset = None
        self.from_scratch = scratch

        if scratch:
            self.model_dir = "./models/cantonese_tokenizer/"

    def preprocess_data(self):
        self.ds = self.ds.filter(lambda x: len(x["text"]) > 100)  # Remove stubs/empty pages

        def tokenize(example):
            return self.tokenizer(example["text"], return_special_tokens_mask=True, truncation=False)

        tokenized_ds = self.ds.map(tokenize, batched=True, remove_columns=["text", "title", "id", "url"])

        # For example, into 512-token chunks
        block_size = 512

        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = (len(concatenated["input_ids"]) // block_size) * block_size
            result = {
                k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated.items()
            }
            return result

        train_dataset, validation_dataset = tokenized_ds["train"].train_test_split(test_size=0.1).values()
        self.lm_dataset = {
            "train": train_dataset.map(group_texts, batched=True),
            "validation": validation_dataset.map(group_texts, batched=True)
        }

    def train(self):
        self.preprocess_data()

        if any({split not in self.lm_dataset for split in ["train", "validation"]}):
            raise ValueError(f"'train' and 'validation' splits must be present in lm_dataset."
                             f"Found: {self.lm_dataset.keys()}")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        config = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,  # 512 + 2
            type_vocab_size=2,
            pad_token_id=self.tokenizer.pad_token_id,
)

        if self.from_scratch:
            model = BertForMaskedLM(config=config)
        else:
            model = BertForMaskedLM.from_pretrained(self.model_dir)

        output_dir_name = f"./{self.lang}-scratch" if self.from_scratch else "./{self.lang}-transfer"

        training_args = TrainingArguments(
            output_dir=output_dir_name,
            overwrite_output_dir=True,
            learning_rate=2e-5,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["validation"],
            data_collator=data_collator,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(output_dir_name)

class CantoPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="yue", model_dir="./models/bert-base-chinese-local", scratch=False):
        super().__init__(lang, model_dir, scratch)
        if not os.path.exists("./data/yue-wiki-full-local"):
            raise FileNotFoundError(
                "Cantonese Wiki dataset not found. Please first run `python download.py --lang=yue`."
            )
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang=yue --model_dir={self.model_dir}`."
            )
        self.ds = load_from_disk("./data/yue-wiki-full-local")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)

    def preprocess_sent_data(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True,
                                  padding="max_length", max_length=128)

        train_dataset, validation_dataset = self.ds["train"].train_test_split(test_size=0.1).values()
        self.lm_dataset = {
            "train": train_dataset.map(tokenize_function, batched=True, remove_columns=["text"]),
            "validation": validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        }

class WuPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="wuu", model_dir="./models/bert-base-chinese-local"):
        super().__init__(lang, model_dir)
        if not os.path.exists("./data/wuu-wiki-local"):
            raise FileNotFoundError(
                "Wu Wiki dataset not found. Please first run `python download.py --lang=wuu`."
            )
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang=wuu --model_dir={self.model_dir}`."
            )
        self.ds = load_from_disk("./data/wuu-wiki-local")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)


def compute_nli_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "preds": preds,
        "labels": labels,
        "accuracy": acc,
        "confusion_matrix": cm.tolist()
    }


class CantoNLIFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir, eval_only=False):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.preprocess_data(eval_only=eval_only)
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
        self.training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

    def preprocess_data(self, eval_only=False):
        nli_data = load_from_disk("./data/yue-nli-local")

        def tokenize_function(examples):
            return self.tokenizer(examples["input_text"], return_special_tokens_mask=True, truncation=True,
                                  padding="max_length", max_length=256)

        def yue_nli_collator(split):
            nested_nli_list = [
                [
              {"input_text": f"{s['anchor']} [SEP] {s['positive']}", "label": 0},
              {"input_text": f"{s['anchor']} [SEP] {s['negative']}", "label": 1}
              ]
                               for s in split]

            return Dataset.from_list([e for s in nested_nli_list for e in s])

        if eval_only:
            test = get_subset(nli_data["test"])
            test_set = yue_nli_collator(test).map(tokenize_function, batched=True)
            print(f"{len(test_set)} test examples.")
            self.finetune_dataset = {
                "train": None,
                "validation": None,
                "test": test_set
            }
        else:
            train, val, test = (
                nli_data["train"],
                nli_data["dev"],
                get_subset(nli_data["test"])
            )

            train_set, val_set, test_set = [
                yue_nli_collator(train).map(tokenize_function, batched=True),
                yue_nli_collator(val).map(tokenize_function, batched=True),
                yue_nli_collator(test).map(tokenize_function, batched=True)
            ]

            print(f"A total of {len(train_set)} training examples,")
            print(f"{len(val_set)} validation examples,")
            print(f"{len(test_set)} test examples.")

            self.finetune_dataset = {
                "train": train_set,
                "validation": val_set,
                "test": test_set
            }

    def eval(self, trainer):
        trainer.compute_metrics = compute_nli_metrics
        metrics = trainer.evaluate(
            eval_dataset=self.finetune_dataset["test"],
        )
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Confusion_matrix: {metrics['eval_confusion_matrix']}")

    def finetune(self):
        if any({split not in self.finetune_dataset for split in ["train", "validation", "test"]}):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["validation"],
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}")
        self.eval(trainer)


class CantoPOSFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
                    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        self.tag2id = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    def preprocess_data(self):
        # Load raw dataset (adjust this path to your actual source)
        raw_data = load_from_disk("./data/yue-pos")  # Should return DatasetDict
        # Expected format: {"train": [{"sentence": [...], "labels": [...]}], ...}

        def align_labels_with_tokens(examples):
            tokenized = self.tokenizer(
                examples["sentence"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128
            )

            labels = []
            for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["sentence"]))):
                word_labels = examples["labels"][i]
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.tag2id.get(word_labels[word_idx], -100))
                        previous_word_idx = word_idx
                    else:
                        label_ids.append(-100)  # Mask subsequent subwords
                labels.append(label_ids)

            tokenized["labels"] = labels
            return tokenized

        # Tokenize and align labels
        tokenized_data = raw_data.map(align_labels_with_tokens, batched=True)
        self.finetune_dataset = {
            "train": tokenized_data["train"],
            "test": tokenized_data.get("test", tokenized_data["train"])
        }

    def finetune(self):
        self.preprocess_data()

        if any(split not in self.finetune_dataset for split in ["train", "test"]):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        model = BertForTokenClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.tag2id),
            id2label=self.id2tag,
            label2id=self.tag2id
        )

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=100,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
        )

        def pos_compute_metrics(pred):
            predictions, labels = pred
            predictions = np.argmax(predictions, axis=-1)

            true_labels = [
                [self.id2tag[l] for (p, l) in zip(pred_row, label_row) if l != -100]
                for pred_row, label_row in zip(predictions, labels)
            ]
            true_preds = [
                [self.id2tag[p] for (p, l) in zip(pred_row, label_row) if l != -100]
                for pred_row, label_row in zip(predictions, labels)
            ]

            # Simple accuracy and F1 (could replace with seqeval)
            flat_preds = [p for row in true_preds for p in row]
            flat_labels = [l for row in true_labels for l in row]

            return {
                "accuracy": accuracy_score(flat_labels, flat_preds),
                "macro_f1": f1_score(flat_labels, flat_preds, average="macro"),
                "micro_f1": f1_score(flat_labels, flat_preds, average="micro"),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=pos_compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}")

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final test accuracy: {metrics['eval_accuracy']}")
        print(f"Final test macro F1: {metrics['eval_macro_f1']}")
        print(f"Final test micro F1: {metrics['eval_micro_f1']}")



class BertForDependencyParsing(BertPreTrainedModel):
    """
    head_classifier: predicts head index in [0..max_length-1] (we map ROOT -> [CLS] position 0)
    rel_classifier: predicts dependency relation label for each token
    """
    def __init__(self, config, num_rel_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Predict head index among max_length token positions
        self.head_classifier = nn.Linear(config.hidden_size, config.max_position_embeddings)
        # Predict relation label
        self.rel_classifier = nn.Linear(config.hidden_size, num_rel_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_head=None,   # [B, L] with indices in [0..L-1], -100 to ignore
        labels_rel=None,    # [B, L] with rel ids, -100 to ignore
        **kwargs
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(outputs.last_hidden_state)  # [B, L, H]

        head_logits = self.head_classifier(seq)  # [B, L, max_pos] (use as [B, L, L] effectively)
        rel_logits  = self.rel_classifier(seq)   # [B, L, R]

        loss = None
        if labels_head is not None and labels_rel is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten over tokens
            head_loss = ce(head_logits.view(-1, head_logits.size(-1)), labels_head.view(-1))
            rel_loss  = ce(rel_logits.view(-1, rel_logits.size(-1)), labels_rel.view(-1))
            loss = head_loss + rel_loss

        # Return a tuple so Trainer hands both logits to compute_metrics
        return {"loss": loss, "logits": (head_logits, rel_logits)}

# ----------------------
# Fine-tuner
# ----------------------
class CantoDEPSFineTuner(CantoPreTrainer):
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)

        # You can expand/adjust to your UD label set (incl. language-specific subtypes like discourse:sp)
        self.dep_labels = [
            "root","nsubj","obj","iobj","obl","vocative","expl","dislocated",
            "advcl","advmod","discourse","aux","cop","mark","nmod","appos",
            "nummod","acl","amod","det","clf","case","conj","cc","fixed",
            "flat","compound","list","parataxis","orphan","goeswith","reparandum",
            "punct","dep","csubj","xcomp","ccomp"
        ]
        self.rel2id = {r: i for i, r in enumerate(self.dep_labels)}
        self.id2rel = {i: r for r, i in self.rel2id.items()}

    def preprocess_data(self):
        """
        Expect dataset with fields:
          - 'sentence': list[str] words
          - 'heads':   list[int] UD heads (0=ROOT, 1..n word indices)
          - 'rels':    list[str] dependency relations, 'root' for head==0
        We align to subwords and:
          - place gold labels only on first subword of each word
          - map head word index -> tokenized sequence index of that head's FIRST subword
          - map ROOT (0) -> [CLS] position index (usually 0)
        """
        raw = load_from_disk("./data/yue-deps")

        def align(examples):
            # Tokenize with word alignment
            enc = self.tokenizer(
                examples["sentence"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_offsets_mapping=False,
            )

            B = len(examples["sentence"])
            labels_head = []
            labels_rel  = []

            for i in range(B):
                word_ids = enc.word_ids(batch_index=i)  # len = seq_len (incl CLS/SEP/PAD)
                words = examples["sentence"][i]
                heads = examples["heads"][i]  # UD heads: 0..len(words)
                rels  = examples["deps"][i]

                # Build map: word_idx -> token_idx of FIRST subword
                # word indices in UD are 1-based; we'll keep that in mind
                first_tok_idx_of_word = {}
                prev_w = None
                for tok_idx, w in enumerate(word_ids):
                    if w is None:
                        continue
                    if w != prev_w:
                        first_tok_idx_of_word[w] = tok_idx
                        prev_w = w

                # Choose a ROOT anchor: map to [CLS] token's position
                # Typically [CLS] is at index 0 with BERT tokenizers
                root_tok_idx = next((i for i, w in enumerate(word_ids) if w is None), 0)

                # Now create aligned labels for each token position
                seq_heads = []
                seq_rels  = []
                prev_w = None
                for tok_idx, w in enumerate(word_ids):
                    if w is None:
                        # Special or padding positions
                        seq_heads.append(-100)
                        seq_rels.append(-100)
                        continue

                    if w != prev_w:
                        # First subword for word w
                        gold_head_word = heads[w]  # if words indexed 0..n-1 in your data, adjust accordingly
                        # If your 'heads' are UD-style (0..n), but our word_ids are 0..n-1,
                        # then gold_head_word==0 means ROOT, otherwise (1..n) -> map to word (gold_head_word-1).
                        if max(heads) <= len(words) and min(heads) == 0:
                            # UD-style 1-based heads
                            if gold_head_word == 0:
                                gold_head_tok = root_tok_idx
                            else:
                                gold_head_tok = first_tok_idx_of_word.get(gold_head_word - 1, root_tok_idx)
                        else:
                            # Already 0-based word indices with -1/None for root? (less common)
                            if gold_head_word < 0:
                                gold_head_tok = root_tok_idx
                            else:
                                gold_head_tok = first_tok_idx_of_word.get(gold_head_word, root_tok_idx)

                        seq_heads.append(gold_head_tok)
                        seq_rels.append(self.rel2id.get(rels[w], -100))
                        prev_w = w
                    else:
                        # Non-first subword -> ignore
                        seq_heads.append(-100)
                        seq_rels.append(-100)

                labels_head.append(seq_heads)
                labels_rel.append(seq_rels)

            enc["labels_head"] = labels_head
            enc["labels_rel"]  = labels_rel
            return enc

        tokenized = raw.map(align, batched=True)
        self.finetune_dataset = {
            "train": tokenized["train"],
            "test": tokenized.get("test", tokenized["train"]),
        }

    def finetune(self):
        self.preprocess_data()

        model = BertForDependencyParsing.from_pretrained(
            self.model_dir,
            num_rel_labels=len(self.rel2id),
        )

        args = TrainingArguments(
            output_dir=f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}",
            overwrite_output_dir=True,
            num_train_epochs=10,
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=50,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="eval_las",
            greater_is_better=True,
        )

        def compute_deps_metrics(eval_pred):
            """
            UAS: pred_head == gold_head
            LAS: pred_head == gold_head AND pred_rel == gold_rel
            Evaluated only where gold labels != -100 (i.e., first subwords).
            """
            preds = eval_pred.predictions
            labels = eval_pred.label_ids

            # Predictions: tuple(head_logits, rel_logits)
            if isinstance(preds, tuple) and len(preds) == 2:
                head_logits, rel_logits = preds
            elif isinstance(preds, dict) and "logits" in preds:
                head_logits, rel_logits = preds["logits"]
            else:
                raise ValueError("Unexpected predictions structure")

            # Labels may be dict or tuple depending on HF version
            if isinstance(labels, dict):
                gold_heads = labels["labels_head"]
                gold_rels  = labels["labels_rel"]
            elif isinstance(labels, (list, tuple)) and len(labels) == 2:
                gold_heads, gold_rels = labels
            else:
                # Some HF versions pass a single array; not our case.
                raise ValueError("Unexpected label_ids structure")

            # Argmax
            pred_heads = np.argmax(head_logits, axis=-1)  # [B, L]
            pred_rels  = np.argmax(rel_logits, axis=-1)   # [B, L]

            gold_heads = np.array(gold_heads)
            gold_rels  = np.array(gold_rels)

            # Valid positions: where gold_rel != -100 (equivalently gold_head != -100)
            valid_mask = gold_rels != -100

            total = valid_mask.sum()
            if total == 0:
                return {"uas": 0.0, "las": 0.0}

            uas_correct = ((pred_heads == gold_heads) & valid_mask).sum()
            las_correct = ((pred_heads == gold_heads) & (pred_rels == gold_rels) & valid_mask).sum()

            uas = float(uas_correct) / float(total)
            las = float(las_correct) / float(total)
            return {"uas": uas, "las": las}

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_deps_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-deps-{self.model_dir.strip('/').split('/')[-1]}")