from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time
import random

def get_attn_time(file):
    fp = open(file, 'r')
    Lines = fp.readlines()
    
    cnt = 0
    res = 0
    for idx, line in enumerate(Lines):
        tmp = float(line)
        res += tmp
        cnt += 1
    return res / cnt

def prepare_time_attn(file):
    fp = open(file, 'r')
    Lines = fp.readlines()

    if len(Lines) == 0:
        return 0
    else:
        cnt = 0
        time = 0
        res = 0
        for idx, line in enumerate(Lines):
            if idx < 18:
                continue
            tmp = float(line)
            time += tmp
            if(idx % 18 == 17):
                res += time
                time = 0
                cnt += 1
        
        return res / cnt

rand_seed = []
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def run_training(Iter):
    training_time = 0
    loss = 0
    for i in range(Iter):
        print(i)
        training_args = TrainingArguments(
                                output_dir = "test-trainer-Test",
                                overwrite_output_dir = True, 
                                evaluation_strategy="no",
                                max_steps = 1,
                                fp16=False,
                                logging_first_step = True,
                                logging_steps = 1,
                                per_device_train_batch_size = 8,
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
        training_time += trainer.state.log_history[1]['train_runtime']
        loss += trainer.state.log_history[0]['loss']

    return training_time, loss

Iter = 20
# enable ATTNChecker
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("2")
# get perpation time
with open("./control/DEBUG.txt", "w") as fr:
    fr.truncate(0)
    fr.write('t')
run_training(Iter+1)

with open("./records/time/attn.txt", "w") as fr:
    fr.truncate(0)

# get true training time
with open("./control/DEBUG.txt", "w") as fr:
    fr.truncate(0)
    fr.write('f')
T, lossLis = run_training(Iter)
    

num_attn_heads = 12
Attn = "./records/time/attn.txt"
prepartion = "./records/time/preparation.txt"

preparation_time = prepare_time_attn(prepartion)
attn_AttnTime = get_attn_time(Attn)*1000 - (preparation_time)
attn_TrainingTime = (T / Iter)*1000 - num_attn_heads*preparation_time
attn_Loss = lossLis / Iter


# disable ATTNChecker
with open("./control/AttnChecker_Mod.txt", "w") as fr:
    fr.truncate(0)
    fr.write("0")
with open("./records/time/attn.txt", "w") as fr:
    fr.truncate(0)

# get training time
with open("./control/DEBUG.txt", "w") as fr:
    fr.truncate(0)
    fr.write('f')
T, lossLis = run_training(Iter)

AttnTime = get_attn_time(Attn)*1000
TrainingTime = (T / Iter)*1000
Loss = lossLis / Iter

print("Attention Mechanism Overhead: ", (attn_AttnTime-AttnTime)/AttnTime)
print("Training Overhead: ", (attn_TrainingTime-TrainingTime)/TrainingTime)

# print("ATTNChecker Attn Time: ", attn_AttnTime)
# print("ATTNChecker Training Time: ", attn_TrainingTime)
print("ATTNChecker Loss: ", attn_Loss)


# print("no ATTNChecker Attn Time: ", AttnTime)
# print("no ATTNChecker Training Time: ", TrainingTime)
print("no ATTNChecker Loss: ", Loss)
