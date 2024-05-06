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
parser.add_argument('-s', '--stride', type=int, default=None,
                    help='Choose the stride for the sliding window dataset.')
parser.add_argument('-dg', '--data_augmentation', type=bool, default=False,
                    help='Choose whether to use data augmentation or not.')

args = parser.parse_args()

if args.stride is not None and args.stride < 0:
    raise ValueError("Stride must be greater than zero.")

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import transformers
import numpy as np
from utils.metric_tracking import MetricsTracking
from utils.dataloader_huggingface import SlidingWindowDataset, CutoffLengthDataset
from datasets import concatenate_datasets


model_checkpoint = "PlanTL-GOB-ES/bsc-bio-ehr-es"
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

tokenizer = AutoTokenizer.from_pretrained(tokenizer_chkp)

assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

if args.stride:
    dataloader = SlidingWindowDataset(max_tokens, tokenizer, ids_to_label, label_to_ids, args.stride)
else:
    dataloader = CutoffLengthDataset(max_tokens, tokenizer, ids_to_label, label_to_ids)

dataset_train = dataloader.get_dataset("../datasets/track1_converted/train/all_train.conll")
dataset_test = dataloader.get_dataset("../datasets/track1_converted/dev/all_dev.conll")

if args.data_augmentation:
    dataset_augmented = dataloader.get_dataset("../datasets/additional/all_train.conll")
    dataset_train = concatenate_datasets([dataset_train, dataset_augmented])

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