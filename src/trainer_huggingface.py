import argparse

parser = argparse.ArgumentParser(
        description='This class is used to train a transformer-based model on for the NER task. Implementation in based on the Huggingface Trainer.')

parser.add_argument('-o', '--output', type=str, default="",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-input', '--input', type=str, default=None,
                    help='Choose the path to the pretrained model. Optional.')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5,
                    help='Choose the learning rate of the model.')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Choose the batch size of the model.')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Choose the epochs of the model.')
parser.add_argument('-l', '--input_length', type=int, default=512,
                    help='Choose the maximum length of the model\'s input layer.')

args = parser.parse_args()

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import transformers
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import torch
from utils.metrics import MetricsTracking


model_checkpoint = "bert-base-multilingual-cased"
tokenizer_chkp = model_checkpoint
model_chkp = model_checkpoint
max_tokens = args.input_length

if args.input:
    model_checkpoint = args.input
    tokenizer_chkp = 'tok_' + model_checkpoint
    model_chkp = 'model_' + model_checkpoint

label_to_ids = {
    'B-ENFERMEDAD': 0,
    'I-ENFERMEDAD': 1,
    'O': 2
}

ids_to_label = {
    0:'B-ENFERMEDAD',
    1:'I-ENFERMEDAD',
    2:'O'
}

label_list = list(ids_to_label.values())

def load_ner_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []
    
    for line in lines:
        line = line.strip()
        if line == "":
            if current_tokens and current_labels:
                tokens.append(current_tokens)
                labels.append(current_labels)
                current_tokens = []
                current_labels = []
        else:
            parts = line.split()
            token = parts[0]
            label = parts[1]
            current_tokens.append(token)
            current_labels.append(label)
    
    if current_tokens and current_labels:
        tokens.append(current_tokens)
        labels.append(current_labels)
    
    df = pd.DataFrame({"tokens": tokens, "labels": labels})
    
    return Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_chkp)

assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

def tokenize_and_preserve_labels(sentence, text_labels, max_tokens):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        if(len(tokenized_sentence)>=max_tokens): #truncate
            return tokenized_sentence, labels

        tokenized_sentence.extend(tokenized_word)

        if label.startswith("B-"):
            labels.extend([label])
            labels.extend([ids_to_label.get(label_to_ids.get(label)+1)]*(n_subwords-1))
        else:
            labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def tokenize_and_align_labels(examples):
    
    t_sen, t_labl = tokenize_and_preserve_labels(examples['tokens'], examples['labels'], max_tokens)

    sen_code = tokenizer.encode_plus(examples['tokens'],
        add_special_tokens=True, # adds [CLS] and [SEP]
        max_length = max_tokens, # maximum tokens of a sentence
        padding='max_length',
        is_split_into_words=True,
        return_attention_mask=True, # generates the attention mask
        truncation = True
        )

    #shift labels (due to [CLS] and [SEP])
    labels = [-100]*max_tokens #-100 is ignore token
    for i, tok in enumerate(t_labl):
        if tok != None and i < max_tokens-1:
            labels[i+1]=label_to_ids.get(tok)

    item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        
    item['entity'] = torch.as_tensor(labels)
    
    return item

def prepare_dataset(dataset):
  tokenized_datasets = dataset.map(tokenize_and_align_labels)

  tokenized_datasets = tokenized_datasets.remove_columns('tokens')
  tokenized_datasets = tokenized_datasets.remove_columns('labels')
  tokenized_datasets = tokenized_datasets.rename_column("entity", "labels")

  return tokenized_datasets

dataset_train = load_ner_dataset("../datasets/track1_converted/train/all_train.conll")
dataset_test = load_ner_dataset("../datasets/track1_converted/dev/all_dev.conll")

dataset_train = prepare_dataset(dataset_train)
dataset_test = prepare_dataset(dataset_test)


model = AutoModelForTokenClassification.from_pretrained(model_chkp, num_labels=len(ids_to_label))

model_name = model_checkpoint.split("/")[-1]
args_train = TrainingArguments(
    f"{model_name}-finetuned-multicardioner",
    evaluation_strategy = "epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_true_predictions = [p for sublist in true_predictions for p in sublist]

    flat_true_labels = [l for sublist in true_labels for l in sublist]

    tracker = MetricsTracking('ENFERMEDAD', tensor_input=False)
    tracker.update(flat_true_predictions, flat_true_labels)

    return tracker.return_avg_metrics()
    
trainer = Trainer(
    model,
    args_train,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

if args.output:
    model.save_pretrained("model_" + args.output)
    tokenizer.save_pretrained("tok_" + args.output)
    print(f"Model has successfully been saved at {args.output}!")