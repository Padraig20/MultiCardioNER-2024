import argparse

parser = argparse.ArgumentParser(
        description='This class is used to train a transformer-based model on for the NER task.')

parser.add_argument('-o', '--output', type=str, default="",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5,
                    help='Choose the learning rate of the model.')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Choose the batch size of the model.')
parser.add_argument('-e', '--epochs', type=int, default=1,
                    help='Choose the epochs of the model.')
parser.add_argument('-opt', '--optimizer', type=str, default='SGD',
                    help='Choose the optimizer to be used for the model: SDG | Adam')
parser.add_argument('-v', '--verbose', type=bool, default=False,
                    help='Choose whether the model should be evaluated after each epoch or only after the training.')
parser.add_argument('-l', '--input_length', type=int, default=128,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-w', '--window_stride', type=int, default=100,
                    help='Choose how large the window stride for attention inside notes should be.')

args = parser.parse_args()

if args.window_stride >= args.input_length:
    raise ValueError("Window stride cannot be larger than or equal to input length.")

from utils.dataloader import Dataloader
from utils.training import train_loop, testing
from utils.models import BertNER
from utils.wandb_logger import WandBLogger

import torch
from torch.optim import SGD
from torch.optim import Adam

#-------MAIN-------#

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

#label_to_ids = {'B-ANAT': 0, 'B-CHEM': 2, 'B-ENFERMEDAD': 4, 'B-PROC': 6, 'I-ANAT': 1, 'I-CHEM': 3, 'I-ENFERMEDAD': 5, 'I-PROC': 7, 'O': 8}
#ids_to_label = {0: 'B-ANAT', 1: 'I-ANAT', 2: 'B-CHEM', 3: 'I-CHEM', 4: 'B-ENFERMEDAD', 5: 'I-ENFERMEDAD', 6: 'B-PROC', 7: 'I-PROC', 8: 'O'}

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

dataloader = Dataloader(label_to_ids, ids_to_label, args.input_length, args.window_stride)

train, val, test = dataloader.load_dataset()

model = BertNER(tokens_dim = len(label_to_ids))

if args.optimizer == 'SGD':
    print("Using SGD optimizer...")
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9)
else:
    print("Using Adam optimizer...")
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

if torch.cuda.is_available():
    print("Using GPU...")
else:
    print("Using CPU...")
    
wandb = WandBLogger(enabled=args.verbose, model=model)
if wandb.enabled:
    wandb.watch(model)

parameters = {
    "model": model,
    "train_dataset": train,
    "eval_dataset" : val,
    "optimizer" : optimizer,
    "batch_size" : args.batch_size,
    "epochs" : args.epochs,
    "type" : "ENFERMEDAD",
    "wandb": wandb
}

train_loop(**parameters, verbose=args.verbose)

testing(model, test, args.batch_size, "ENFERMEDAD")

#save model if wanted
if args.output:
    torch.save(model.state_dict(), args.output)
    print(f"Model has successfully been saved at {args.output}!")

wandb.finish()