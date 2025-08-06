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
)
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from utils import get_subset


class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/bert-base-chinese-local"):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.model_dir = model_dir
        self.tokenized_ds = None
        self.lm_dataset = None

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

        model = BertForMaskedLM.from_pretrained(self.model_dir)

        training_args = TrainingArguments(
            output_dir=f"./{self.lang}-pretrain",
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
        trainer.save_model(f"./{self.lang}-pretrain")

class CantoPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="yue", model_dir="./models/bert-base-chinese-local"):
        super().__init__(lang, model_dir)
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

        def compute_metrics(pred):
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
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-pos-{self.model_dir.strip('/').split('/')[-1]}")

        metrics = trainer.evaluate(self.finetune_dataset["test"])
        print(f"Final test accuracy: {metrics['eval_accuracy']}")
        print(f"Final test macro F1: {metrics['eval_macro_f1']}")
        print(f"Final test macro F1: {metrics['eval_micro_f1']}")

