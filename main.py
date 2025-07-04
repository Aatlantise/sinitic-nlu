from datasets import load_dataset, load_from_disk
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

class SiniticPreTrainer:
    def __init__(self, lang=""):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
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

        model = BertForMaskedLM.from_pretrained("./bert-base-chinese-local")

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
    def __init__(self, lang="yue"):
        super().__init__(lang)
        self.ds = load_from_disk("./yue-wiki-local")
        self.tokenizer = BertTokenizerFast.from_pretrained("./bert-base-chinese-local")

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
    def __init__(self, lang="wuu"):
        super().__init__(lang)
        self.ds = load_dataset("wikimedia/wikipedia", "20231101.wuu")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

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

        self.lm_dataset = tokenized_ds.map(group_texts, batched=True)
