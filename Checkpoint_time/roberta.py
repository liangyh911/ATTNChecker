from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time
import random

import os

# torch.set_default_device('cuda')
rand_seed = []
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "roberta-base"
# checkpoint = "./ROBERTAABFT8/checkpoint-1"
# checkpoint = "./ROBERTAABFT/checkpoint-1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# cnt = 0
# lossLis = []

def get_home_directory_with_expanduser():
    return os.path.expanduser("~")

cnt = 0
lossLis = 0
# T = 0
training_time = 0

save_time = 0
load_time = 0

homePath = get_home_directory_with_expanduser()
path = "./Checkpoint_time/Checkpoint"

# Iter = 5530
Iter = 20

# disable ATTNChecker
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("0")

for i in range(Iter):
    print(i)
    training_args = TrainingArguments(
                            output_dir = "test-trainer-Test",
                            # output_dir = path,
                            overwrite_output_dir = True, 
                            evaluation_strategy="no",
                            # num_train_epochs = 4, 
                            max_steps = 1,
                            fp16=False,
                            # logging_dir = "test-trainer/logs", 
                            logging_first_step = True,
                            # logging_steps = 459,
                            per_device_train_batch_size = 8,
                            save_steps = 1,
                            # seed = i
    )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
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

fr =  open("./ABFT_running_time/robertaCache.txt", "r")
Lines = fr.readlines()
training_time = float(Lines[0])
# attn_training_time = float(Lines[1])
checkpoint_time = training_time + (save_time / Iter + load_time / Iter)*1000

print("Overhead of Checkpointing: ",(checkpoint_time-training_time)/training_time)