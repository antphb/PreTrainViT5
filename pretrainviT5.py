from huggingface_hub import login
import torch
import json
import multiprocessing
import argparse

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,Seq2SeqTrainer,Seq2SeqTrainingArguments,DataCollatorForSeq2Seq)

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--token_huggingface', type=str, required=True, default='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')

parser.add_argument('--model_name', type=str, default='VietAI/vit5-large')

parser.add_argument('--path_data', type=str, default='/home/bbsw/Desktop/Model_nba/UIT/translated_data/data_vi.json')
parser.add_argument('--train_batch_size', type=int, default=1)
parser.add_argument('--valid_batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-05)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

parser.add_argument('--output_dir', type=str, default='pretrain-vit5-large')


args = parser.parse_args()

MAX_LEN = args.max_len
TRAIN_BATCH_SIZE = args.train_batch_size
VALID_BATCH_SIZE = args.valid_batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
TOKEN_HUGGINGFACE = args.token_huggingface
login(token=TOKEN_HUGGINGFACE)
model_name = args.model_name
path_data = args.path_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

def gen_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for i in data:
            try:
                if not isinstance(i['content'], str):
                    continue
                context = " ".join(i['content'].split())
                _data = {
                    'content': context
                }
                f.write(json.dumps(_data, ensure_ascii=False) + '\n')
            except:
                print(i)

def tokenize(examples):
    return tokenizer(examples["content"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= MAX_LEN:
        total_length = (total_length // MAX_LEN) * MAX_LEN
    result = {
        k: [t[i : i + MAX_LEN] for i in range(0, total_length, MAX_LEN)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = [x[-10:] for x in result["input_ids"].copy()]
    result["labels"]=result["input_ids"].copy()
    return result

if __name__=="__main__":
    num_proc = multiprocessing.cpu_count()
    with open(path_data, encoding='utf-8') as f:
        data = json.load(f)
    train_data, test_data = train_test_split(data, test_size=0.05, random_state=42)

    gen_data('train_data.jsonl', train_data)
    gen_data('test_data.jsonl', test_data)

    train_data = load_dataset('json', data_files='train_data.jsonl')['train']
    val_data = load_dataset('json', data_files='test_data.jsonl')['train']

    raw_datasets = DatasetDict(
    {
        "train": train_data,
        "valid": val_data
    })

    tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names, num_proc=num_proc)

    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    tokenized_datasets = tokenized_datasets.shuffle(seed=128)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = "steps",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        eval_steps=200,
        logging_steps=200,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        # gradient_accumulation_steps="auto",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    model.config.use_cache=False
    trainer.train()
    trainer.push_to_hub()
