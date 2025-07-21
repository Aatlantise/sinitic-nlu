from datasets import load_from_disk, Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback, BertForSequenceClassification,
)
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

class SiniticPreTrainer:
    def __init__(self, lang="", model_dir="./models/bert-base-chinese-local"):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.model_dir = model_dir
        self.tokenized_ds = None
        self.lm_dataset = None

    def preprocess_data(self):
        pass

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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        trainer.save_model(f"./{self.lang}-pretrain")

class CantoPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="yue", model_dir="./models/bert-base-chinese-local"):
        super().__init__(lang, model_dir)
        if not os.path.exists("./data/yue-wiki-local"):
            raise FileNotFoundError(
                "Cantonese Wiki dataset not found. Please first run `python download.py --lang=yue`."
            )
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model directory {self.model_dir} not found."
                f"Please first run `python download.py --lang=yue --model_dir={self.model_dir}`."
            )
        self.ds = load_from_disk("./data/yue-wiki-local")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)

    def preprocess_data(self):
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
    def __init__(self, lang, model_dir):
        super().__init__(lang, model_dir)
        self.finetune_dataset = None

    def preprocess_data(self):
        nli_data = load_from_disk("./data/yue-nli-local")

        def tokenize_function(examples):
            return self.tokenizer(examples["input_text"], return_special_tokens_mask=True, truncation=True,
                                  padding="max_length", max_length=256)

        train, val, test = (
            nli_data["train"],
            nli_data["dev"],
            nli_data["test"]
        )

        def yue_nli_collator(split):
            nested_nli_list = [
                [
              {"input_text": f"{s['anchor']} [SEP] {s['positive']}", "label": 0},
              {"input_text": f"{s['anchor']} [SEP] {s['negative']}", "label": 1}
              ]
                               for s in split]

            return Dataset.from_list([e for s in nested_nli_list for e in s])

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

    def finetune(self):
        self.preprocess_data()

        if any({split not in self.finetune_dataset for split in ["train", "validation", "test"]}):
            raise ValueError(f"'train' and 'validation' splits must be present in finetune_dataset."
                             f"Found: {self.finetune_dataset.keys()}")

        model = BertForSequenceClassification.from_pretrained(self.model_dir)

        training_args = TrainingArguments(
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["validation"],
        )

        trainer.train()
        trainer.save_model(f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}")

        trainer.compute_metrics = compute_nli_metrics
        metrics = trainer.evaluate(
            eval_dataset=self.finetune_dataset["test"],
            )
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Confusion_matrix: {metrics['eval_confusion_matrix']}")
