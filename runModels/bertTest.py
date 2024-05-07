from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import time
import random

# torch.set_default_device('cuda')
rand_seed = []
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
# checkpoint = "./test-trainer-TestInf/checkpoint-1"
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

cnt = 0
lossLis = 0
T = 0

Iter = 1
for i in range(Iter):
    print(i)
    training_args = TrainingArguments(
                            output_dir = "test-trainer-Test",
                            overwrite_output_dir = True, 
                            evaluation_strategy="no",
                            # num_train_epochs = 1, 
                            max_steps = 1,
                            fp16=False,
                            # logging_dir = "test-trainer/logs", 
                            logging_first_step = True,
                            logging_steps = 1,
                            # per_device_train_batch_size = 16,
                            # save_steps = 1,
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

    start = time.perf_counter()

    # random.seed(i)
    trainer.train()

    T += (time.perf_counter() - start)
    lossLis += trainer.state.log_history[0]['loss']
    # trainer.evaluate()
    # print(f'training time, {(time.perf_counter() - start):.1f} s')
    # if trainer.state.log_history[0]['loss'] == 0:
    #     cnt += 1
    #     print("Current cnt: ", cnt)

print("Time: ", T / Iter)
print("Loss: ", lossLis / Iter)


# print('There are ', cnt, 'Injections make loss = 0.')
# print('P(loss = 0) = ', cnt/Iter)
