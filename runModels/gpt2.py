from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
    # return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# checkpoint = "distilbert/distilbert-base-uncased"
checkpoint = 'gpt2'
# checkpoint = 'microsoft/phi-1'
# checkpoint = 'bert-base-uncased'

# data = load_dataset("imdb")
data = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# tokenized_imdb = data.map(preprocess_function, batched=True)
tokenized_imdb = data.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}

cnt = 0
lossLis = 0
T = 0
# iter = 5453
iter = 1
for i in range(iter):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, 
        num_labels=2, 
        # id2label=id2label, 
        # label2id=label2id, 
        # attn_implementation="eager", 
    )
    model.config.pad_token_id = model.config.eos_token_id       

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        per_device_train_batch_size=8,
        # num_train_epochs=1,
        max_steps = 1,
        fp16=False,
        evaluation_strategy="no",
        logging_first_step = True,
        logging_steps = 1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    trainer.train()
    T += (time.perf_counter() - start)
    lossLis += trainer.state.log_history[0]['loss']
    # if trainer.state.log_history[0]['loss'] == 0:
    #     cnt += 1
    # print("Current cnt: ", cnt)

print("Time: ", T / iter)
print("Loss: ", lossLis / iter)