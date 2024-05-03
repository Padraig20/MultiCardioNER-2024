import argparse

parser = argparse.ArgumentParser(
        description='This class is used to pre-train a transformer-based model on admission notes.')

parser.add_argument('-o', '--output', type=str, default=None,
                    help='Choose where to save the model after pre-training.')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
                    help='Choose the learning rate of the model.')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Choose the epochs of the model.')
parser.add_argument('-l', '--input_length', type=int, default=512,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01,
                    help='Choose the weight decay of the model during training.')
parser.add_argument('-mlm', '--mlm_probability', type=float, default=0.15,
                    help='Choose the probability of masking tokens during training.')

args = parser.parse_args()

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import math

from utils.dataloader import Admission_Notes_Dataset
from utils.models import BertMLM, get_tokenizer

model = BertMLM()
tokenizer = get_tokenizer()

with open("../datasets/admission_notes/es", "r") as f:
    texts = f.readlines()

encodings = tokenizer(texts,
                      add_special_tokens=True,
                      truncation=True,
                      padding='max_length',
                      max_length=args.input_length,
                      return_tensors="pt")
dataset = Admission_Notes_Dataset(encodings)

train_size = int(0.9 * len(dataset))
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

training_args = TrainingArguments(
    output_dir=args.output,
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if args.output:
    model.save_pretrained("model_" + args.output)
    tokenizer.save_pretrained("tok_" + args.output)
    print(f"Model saved to {args.output}.")