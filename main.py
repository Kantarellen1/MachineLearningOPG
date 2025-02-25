import torch
import json
from datasets import Dataset
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq


with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

train_data = dataset["train"]
test_data = dataset["test"]


train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)


tokenizer = PreTrainedTokenizerFast(tokenizer_file="danish_tokenizer.model")


def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)


model = T5ForConditionalGeneration.from_pretrained("t5-small")


training_args = TrainingArguments(
    output_dir="./results",  
    evaluation_strategy="epoch",  
    learning_rate=3e-5,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    num_train_epochs=3,  
    weight_decay=0.01,  
    save_total_limit=2,  
    logging_dir="./logs",  
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()


model.save_pretrained("danish_summarization_model")
tokenizer.save_pretrained("danish_summarization_model")


def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


print(summarize("Danmark investerer massivt i grøn energi for at bekæmpe klimaforandringer."))

