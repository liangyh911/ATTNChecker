from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time
import os

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
    # return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

checkpoint = 'EleutherAI/gpt-neo-125M'
# checkpoint = "./NEOABFT8/checkpoint-1"

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

def get_home_directory_with_expanduser():
    return os.path.expanduser("~")

# cnt = 0
loss = 0
# T = 0
training_time = 0

save_time = 0
load_time = 0

homePath = get_home_directory_with_expanduser()
path = "./Checkpoint_time/Checkpoint"

# iter = 5453
Iter = 20

# disable ATTNChecker
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("0")

for i in range(Iter):
    print(i)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, 
        # id2label=id2label, 
        # label2id=label2id, 
        attn_implementation="eager", 
    )
    model.config.pad_token_id = model.config.eos_token_id       

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        # output_dir = path,
        per_device_train_batch_size=8,
        # num_train_epochs=3,
        max_steps = 1,
        fp16=False,
        evaluation_strategy="no",
        logging_first_step = True,
        # logging_steps = 459,
        save_steps = 1,
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
    
    trainer.train()

    start = time.perf_counter()
    trainer.save_model(path)
    save_time += (time.perf_counter() - start)

    start = time.perf_counter()
    re_model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
    load_time += (time.perf_counter() - start)


# print("Save Time: ", (save_time / Iter)*1000)
# print("Load Time: ", (load_time / Iter)*1000)
# print("Saving and Loading time: ", (save_time / Iter + load_time / Iter)*1000)

fr =  open("./ABFT_running_time/gptneoCache.txt", "r")
Lines = fr.readlines()
training_time = float(Lines[0])
# attn_training_time = float(Lines[1])
checkpoint_time = training_time + (save_time / Iter + load_time / Iter)*1000

print("Overhead of Checkpointing: ",(checkpoint_time-training_time)/training_time)
