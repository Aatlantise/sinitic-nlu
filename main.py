from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertForMaskedLM,
    BertForPreTraining,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

class SiniticPreTrainer:
    def __init__(self, lang=""):
        self.ds = None
        self.tokenizer = None
        self.lang = lang
        self.tokenized_ds = None
        self.lm_dataset = None

    def prerocess_data(self):
        pass

    def train(self):
        self.preprocess_data()

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        model = BertForMaskedLM.from_pretrained("bert-base-chinese")

        training_args = TrainingArguments(
            output_dir=f"./{self.lang}-pretrain",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            report_to="tensorboard"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(f"./{self.lang}-pretrain")

class CantoPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="yue"):
        super().__init__(lang)
        self.ds = load_dataset("R5dwMg/zh-wiki-yue-long")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    def preprocess_data(self):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True,
                                  padding="max_length", max_length=128)

        self.lm_dataset = self.ds.map(tokenize_function, batched=True)

class WuPreTrainer(SiniticPreTrainer):
    def __init__(self, lang="wuu"):
        super().__init__(lang)
        self.ds = load_dataset("wikimedia/wikipedia", "20231101.wuu", split="train")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    def preprocess(self):
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
