from utils.dataloader import Dataloader
from utils.models import BertNER
from utils.metric_tracking import MetricsTracking
from utils.wandb_logger import WandBLogger

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

def train_loop(model, train_dataset, eval_dataset, optimizer, batch_size, epochs, type, verbose=True):
    """
    Usual training loop, including training and evaluation.

    Parameters:
    model (BertNER): Model to be trained.
    train_dataset (Custom_Dataset): Dataset used for training.
    eval_dataset (Custom_Dataset): Dataset used for testing.
    optimizer (torch.optim): Optimizer used, usually SGD or Adam.
    batch_size (int): Batch size used during training.
    epochs (int): Number of epochs used for training.
    verbose (bool): Whether the model should be evaluated after each epoch or not.

    Returns:
    tuple:
        - train_res (dict): A dictionary containing the results obtained during training.
        - test_res (dict): A dictionary containing the results obtained during testing.
    """

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    wandb = WandBLogger(enabled=verbose, model=model)
    if wandb.enabled:
        wandb.watch(model)

    #training
    for epoch in range(epochs):

        train_metrics = MetricsTracking(type)
        
        log_dict = dict()

        model.train() #train mode

        for train_data in tqdm(train_dataloader):

            train_label = train_data['entity'].to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()

            output = model(input_id, mask, train_label)
            loss, logits = output.loss, output.logits
            predictions = logits.argmax(dim=-1)

            #compute metrics
            train_metrics.update(predictions, train_label, loss.item())

            loss.backward()
            optimizer.step()

        if verbose:
            model.eval() #evaluation mode

            eval_metrics = MetricsTracking(type)

            with torch.no_grad():

                for eval_data in eval_dataloader:

                    eval_label = eval_data['entity'].to(device)
                    mask = eval_data['attention_mask'].squeeze(1).to(device)
                    input_id = eval_data['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask, eval_label)
                    loss, logits = output.loss, output.logits

                    predictions = logits.argmax(dim=-1)

                    eval_metrics.update(predictions, eval_label, loss.item())

            train_results = train_metrics.return_avg_metrics(len(train_dataloader))
            eval_results = eval_metrics.return_avg_metrics(len(eval_dataloader))
            
            log_dict.update({
                'train/f1_avg': train_results['avg_f1_score'],
                'train/f1_strict': train_results['strict']['f1_score'],
                'train/f1_ent_type': train_results['ent_type']['f1_score'],
                'train/f1_partial': train_results['partial']['f1_score'],
                'train/f1_exact': train_results['exact']['f1_score']
            })
            
            log_dict.update({
                'eval/f1_avg': eval_results['avg_f1_score'],
                'eval/f1_strict': eval_results['strict']['f1_score'],
                'eval/f1_ent_type': eval_results['ent_type']['f1_score'],
                'eval/f1_partial': eval_results['partial']['f1_score'],
                'eval/f1_exact': eval_results['exact']['f1_score']
            })
            
            wandb.log(log_dict)

            print(f"Epoch {epoch+1} of {epochs} finished!")
            print(f"TRAIN\nMetrics {train_results}\n")
            print(f"VALIDATION\nMetrics {eval_results}\n")

    if not verbose:
        model.eval() #evaluation mode

        eval_metrics = MetricsTracking(type)

        with torch.no_grad():

            for eval_data in eval_dataloader:

                eval_label = eval_data['entity'].to(device)
                mask = eval_data['attention_mask'].squeeze(1).to(device)
                input_id = eval_data['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask, eval_label)
                loss, logits = output.loss, output.logits

                predictions = logits.argmax(dim=-1)

                eval_metrics.update(predictions, eval_label, loss.item())

        train_results = train_metrics.return_avg_metrics(len(train_dataloader))
        eval_results = eval_metrics.return_avg_metrics(len(eval_dataloader))
        
        log_dict.update({
            'train/f1_avg': train_results['avg_f1_score'],
            'train/f1_strict': train_results['strict']['f1_score'],
            'train/f1_ent_type': train_results['ent_type']['f1_score'],
            'train/f1_partial': train_results['partial']['f1_score'],
            'train/f1_exact': train_results['exact']['f1_score']
        })
            
        log_dict.update({
            'eval/f1_avg': eval_results['avg_f1_score'],
            'eval/f1_strict': eval_results['strict']['f1_score'],
            'eval/f1_ent_type': eval_results['ent_type']['f1_score'],
            'eval/f1_partial': eval_results['partial']['f1_score'],
            'eval/f1_exact': eval_results['exact']['f1_score']
        })
        
        wandb.log(log_dict)

        print(f"Epoch {epoch+1} of {epochs} finished!")
        print(f"TRAIN\nMetrics {train_results}\n")
        print(f"VALIDATION\nMetrics {eval_results}\n")
    
    wandb.finish()

    return train_results, eval_results

def testing(model, test_dataset, batch_size, type):
    """
    Function for testing a trained model.

    Parameters:
    model (BertNER | BioBertNER): Model to be tested
    train_dataset (Custom_Dataset): Dataset used for testing
    batch_size (int): Batch size used during training.

    Returns:
    tuple:
        - test_res (dict): A dictionary containing the results obtained during testing.
    """

    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval() #evaluation mode

    test_metrics = MetricsTracking(type)

    with torch.no_grad():

        for test_data in test_dataloader:

            test_label = test_data['entity'].to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)
            input_id = test_data['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask, test_label)
            loss, logits = output.loss, output.logits

            predictions = logits.argmax(dim=-1)

            test_metrics.update(predictions, test_label, loss.item())

        test_results = test_metrics.return_avg_metrics(len(test_dataloader))

        print(f"TEST\nMetrics {test_results}\n")

    return test_results
