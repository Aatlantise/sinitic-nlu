from datasets import load_dataset
from transformers import BertForSequenceClassification, BertForMaskedLM, BertForPreTraining, BertTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling

ds = load_dataset("R5dwMg/zh-wiki-yue-long")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=128)

tokenized_ds = ds.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

model = BertForMaskedLM.from_pretrained("bert-base-chinese")

training_args = TrainingArguments(
    output_dir="./canto-pretrain",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./canto-pretrain")