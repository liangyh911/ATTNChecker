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
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

checkpoint = 'EleutherAI/gpt-neo-125M'
data = load_dataset("glue", "mrpc")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

tokenized_imdb = data.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

cnt = 0
lossLis = 0
T = 0
training_time = 0
save_time = 0
load_time = 0

path = "./ABFT_epoch_loss/ckps/GPTneoABFT"

def abft_save_ckp(checkpoint, path):
    training_args = TrainingArguments(
                        # output_dir = "test-trainer-Test",
                        output_dir = path,
                        overwrite_output_dir = True, 
                        evaluation_strategy="no",
                        # num_train_epochs = 3, 
                        max_steps = 1,
                        fp16=False,
                        # logging_dir = "test-trainer/logs", 
                        # logging_first_step = True,
                        # logging_steps = 1,
                        # logging_strategy="epoch",
                        per_device_train_batch_size = 8,
                        save_steps = 1,
                        # seed = i
    )
    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2, 
            # id2label=id2label, 
            # label2id=label2id, 
            attn_implementation="eager", 
        )
    model.config.pad_token_id = model.config.eos_token_id

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def run_from_ckp(checkpoint):
    training_args = TrainingArguments(
                        output_dir = "test-trainer-Test",
                        # output_dir = path,
                        overwrite_output_dir = True, 
                        evaluation_strategy="no",
                        num_train_epochs = 3, 
                        # max_steps = 1,
                        fp16=False,
                        # logging_dir = "test-trainer/logs", 
                        # logging_first_step = True,
                        # logging_steps = 1,
                        logging_strategy="epoch",
                        per_device_train_batch_size = 8,
                        # save_steps = 1,
                        # seed = i
    )
    model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2, 
            # id2label=id2label, 
            # label2id=label2id, 
            attn_implementation="eager", 
        )
    model.config.pad_token_id = model.config.eos_token_id

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer.state.log_history

# enable ATTNChecker and save ckp
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("2")
with open("./control/DEBUG.txt", "w") as fr:
    fr.truncate(0)
    fr.write('f')

abft_save_ckp(checkpoint, path)

checkpoint = "./ABFT_epoch_loss/ckps/GPTneoABFT/checkpoint-1"

# disable ATTNChecker, and training from ckp
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("0")

attnchk_log = run_from_ckp(checkpoint)

checkpoint = "EleutherAI/gpt-neo-125M"
baseline_log = run_from_ckp(checkpoint)

attnchk_loss = [e['loss'] for e in attnchk_log[:-1]]
baseline_loss = [e['loss'] for e in baseline_log[:-1]]

print("Loss of ATTNChecker: ")
print("1st epoch: ", attnchk_loss[0], ", 2nd epoch: ", attnchk_loss[1], ", 3rd epoch: ", attnchk_loss[2])

print("Loss of Baseline: ")
print("1st epoch: ", baseline_loss[0], ", 2nd epoch: ", baseline_loss[1], ", 3rd epoch: ", baseline_loss[2])

# print("Loss of ATTNChecker: ", [e['loss'] for e in attnchk_log[:-1]])
# print("Loss of Baseline   : ", [e['loss'] for e in baseline_log[:-1]])
# train_time = [t for e in log.]

