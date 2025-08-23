from datasets import load_from_disk, Dataset, load_dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback, BertForSequenceClassification,
    BertForTokenClassification
)
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
import evaluate

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
            output_dir=f"/home/yorkng/scratch/sinitic-nlu/{self.lang}-pretrain",
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
        trainer.save_model(f"/home/yorkng/scratch/sinitic-nlu/{self.lang}-pretrain")

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
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
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


class CantoTokenClassificationFineTuner(CantoNLIFineTuner):
    def __init__(self, lang="yue", model_dir="./bert-base-chinese-local"):
        super().__init__(lang, model_dir)

    def preprocess_data(self):
        nlu_data = load_from_disk('./data/nlptea_dataset')['train']
        k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
        indices = list(k_fold.split(np.arange(len(nlu_data))))

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples[f"cantonese_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        self.finetune_dataset = []

        for fold, (indices_train, indices_test) in enumerate(indices):
            train_set = nlu_data.select(indices_train)
            valid_set = nlu_data.select(indices_test)

            train_set = train_set.map(tokenize_and_align_labels, batched=True)
            valid_set = valid_set.map(tokenize_and_align_labels, batched=True)

            self.finetune_dataset.append({
                "train": train_set,
                "validation": valid_set
            })

    def finetune(self):
        self.preprocess_data()

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        seqeval = evaluate.load("seqeval")
        label_list = ['Chinese', 'Cantonese']
        id2label = {
            0: 'Chinese',
            1: 'Cantonese'
        }
        label2id = {
            'Chinese': 0,
            'Cantonese': 1
        }

        def compute_nlu_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            _, _, f1_scores, _ = precision_recall_fscore_support(
                [l for sublist in true_labels for l in sublist],
                [p for sublist in true_predictions for p in sublist],
                labels=["Cantonese"]
            )
            f1_positive = f1_scores[0]

            return {
                "f1_positive": f1_positive,
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=1e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=50,
            report_to="tensorboard",
        )

        cross_validation_results = {
            'f1_positive': [],
            'f1': [],
            'accuracy': [],
        }

        for fold, dataset in enumerate(self.finetune_dataset):
            print(f"Training on fold {fold + 1}/{len(self.finetune_dataset)}")
            model = BertForTokenClassification.from_pretrained(
                self.model_dir,
                num_labels=2,
                id2label=id2label,
                label2id=label2id
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_nlu_metrics,
            )

            trainer.train()
            # trainer.save_model(f"./models/{self.lang}-nlu-{[f for f in self.model_dir.split('/') if f][-1]}-fold-{fold}")

            metrics = trainer.evaluate(
                eval_dataset=dataset["validation"],
            )
            print(metrics)

            cross_validation_results['accuracy'].append(metrics['eval_accuracy'])
            cross_validation_results['f1_positive'].append(metrics['eval_f1_positive'])
            cross_validation_results['f1'].append(metrics['eval_f1'])
            print(f"Fold {fold + 1} - Accuracy: {metrics['eval_accuracy']}")
            print(f"Fold {fold + 1} - F1: {metrics['eval_f1']}")
            print(f"Fold {fold + 1} - F1 Positive: {metrics['eval_f1_positive']}")

        print("Cross-validation results:")
        print(f"Average Accuracy: {np.mean(cross_validation_results['accuracy'])}")
        print(f"Average F1: {np.mean(cross_validation_results['f1'])}")
        print(f"Average F1 Positive: {np.mean(cross_validation_results['f1_positive'])}")

class CantoAcceptabilityFineTuner(CantoNLIFineTuner):
    def __init__(self, lang="yue", model_dir="./bert-base-chinese-local"):
        super().__init__(lang, model_dir)

    def preprocess_data(self):
        data = load_from_disk('data/acceptability-dataset-2')
        data = data.shuffle(seed=42)

        def tokenize(example):
            return self.tokenizer(example["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=128)

        self.finetune_dataset = []
        train_set, temp_set = data.train_test_split(test_size=0.1, seed=42).values()
        valid_set, test_set = temp_set.train_test_split(test_size=0.5, seed=42).values()

        train_set = train_set.map(tokenize, batched=True, remove_columns=["text"])
        valid_set = valid_set.map(tokenize, batched=True, remove_columns=["text"])
        test_set = test_set.map(tokenize, batched=True, remove_columns=["text"])

        self.finetune_dataset = {
            "train": train_set,
            "validation": valid_set,
            "test": test_set
        }

    def finetune(self):
        self.preprocess_data()
        model = BertForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=3,
            id2label={0: "unacceptable", 1: "acceptable", 2: "mix"},
            label2id={"unacceptable": 0, "acceptable": 1, "mix": 2}
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)

            acc = accuracy_score(labels, preds)
            cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
            macro_f1 = f1_score(labels, preds, average='macro')
            weighted_f1 = f1_score(labels, preds, average='weighted')

            # Get per-class precision, recall, f1
            _, _, f1s, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1])
            f1_positive = f1s[1]  # label=1

            return {
                "accuracy": acc,
                "confusion_matrix": cm.tolist(),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                # "f1_positive": f1_positive  # new key
            }

        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in model.bert.pooler.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        training_args = TrainingArguments(
            output_dir=f"./models/{self.lang}-acceptability2-{[f for f in self.model_dir.split('/') if f][-1]}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            optim="adamw_torch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=20,
            report_to="tensorboard",
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            # metric_for_best_model="eval_f1_positive",
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_steps=1000,
        )
        seqeval = evaluate.load("seqeval")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.finetune_dataset["train"],
            eval_dataset=self.finetune_dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()

        metrics = trainer.evaluate(
            eval_dataset=self.finetune_dataset["test"],
        )
        print(f"Accuracy: {metrics['eval_accuracy']}")
        print(f"Macro F1: {metrics['eval_macro_f1']}")
        # print(f"Positive F1: {metrics['eval_f1_positive']}")
