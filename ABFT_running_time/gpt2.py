from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time

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
            if idx < 12:
                continue
            tmp = float(line)
            time += tmp
            if(idx % 12 == 11):
                res += time
                time = 0
                cnt += 1
        
        return res / cnt

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
    # return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

checkpoint = 'gpt2'

data = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

tokenized_imdb = data.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def run_training(Iter):
    training_time = 0
    loss = 0
    for i in range(Iter):
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, 
            num_labels=2, 
        )
        model.config.pad_token_id = model.config.eos_token_id       

        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            per_device_train_batch_size=8,
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

        trainer.train()
        training_time += trainer.state.log_history[1]['train_runtime']
        loss += trainer.state.log_history[0]['loss']

    return training_time, loss


Iter = 20
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

with open("./ABFT_running_time/gpt2Cache.txt", "a") as fr:
   fr.truncate(0)
   fr.write(str(TrainingTime))
#    fr.write("\n")
#    fr.write(str(attn_TrainingTime))