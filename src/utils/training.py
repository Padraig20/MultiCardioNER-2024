from utils.metric_tracking import MetricsTracking

import torch
from torch.utils.data import DataLoader
from utils.metrics import FocalLoss

from tqdm import tqdm

def train_loop(model, train_dataset, eval_dataset, optimizer, batch_size, epochs, type, wandb, verbose=True):
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
    loss_function = FocalLoss()

    #training
    for epoch in range(epochs):

        train_metrics = MetricsTracking(type)
        
        train_loss_arr = []
        
        log_dict = dict()

        model.train() #train mode

        for train_data in tqdm(train_dataloader):

            train_label = train_data['entity'].to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()

            output = model(input_id, attention_mask=mask) # do not pass train_label here
            logits = output.logits

            loss = loss_function(logits.view(-1, logits.size(-1)), train_label.view(-1))

            train_loss_arr.append(loss.item())

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            train_metrics.update(predictions, train_label)
            
        if verbose:
            model.eval() #evaluation mode

            eval_metrics = MetricsTracking(type)
            
            eval_loss_arr = []

            with torch.no_grad():

                for eval_data in eval_dataloader:

                    eval_label = eval_data['entity'].to(device)
                    mask = eval_data['attention_mask'].squeeze(1).to(device)
                    input_id = eval_data['input_ids'].squeeze(1).to(device)

                    output = model(input_id, attention_mask=mask)
                    logits = output.logits

                    loss = loss_function(logits.view(-1, logits.size(-1)), eval_label.view(-1))
                    
                    eval_loss_arr.append(loss.item())

                    predictions = logits.argmax(dim=-1)

                    eval_metrics.update(predictions, eval_label)

            train_results = train_metrics.return_avg_metrics()
            eval_results = eval_metrics.return_avg_metrics()
            
            log_dict.update({
                'train/f1_avg': train_results['avg_f1_score'],
                'train/f1_strict': train_results['strict']['f1_score'],
                'train/f1_ent_type': train_results['ent_type']['f1_score'],
                'train/f1_partial': train_results['partial']['f1_score'],
                'train/f1_exact': train_results['exact']['f1_score'],
                'train/loss': sum(train_loss_arr)/len(train_loss_arr)
            })
            
            log_dict.update({
                'eval/f1_avg': eval_results['avg_f1_score'],
                'eval/f1_strict': eval_results['strict']['f1_score'],
                'eval/f1_ent_type': eval_results['ent_type']['f1_score'],
                'eval/f1_partial': eval_results['partial']['f1_score'],
                'eval/f1_exact': eval_results['exact']['f1_score'],
                'eval/loss': sum(eval_loss_arr)/len(eval_loss_arr)
            })
            
            wandb.log(log_dict)

            print(f"Epoch {epoch+1} of {epochs} finished!")
            print(f"TRAIN\nMetrics {train_results}\n")
            print(f"VALIDATION\nMetrics {eval_results}\n")
            print(f"TRAIN\nLoss {sum(train_loss_arr)/len(train_loss_arr)}\n")
            print(f"VALIDATION\nLoss {sum(eval_loss_arr)/len(eval_loss_arr)}\n")

    if not verbose:
        model.eval() #evaluation mode

        eval_metrics = MetricsTracking(type)
        
        eval_loss_arr = []

        with torch.no_grad():

            for eval_data in eval_dataloader:

                eval_label = eval_data['entity'].to(device)
                mask = eval_data['attention_mask'].squeeze(1).to(device)
                input_id = eval_data['input_ids'].squeeze(1).to(device)

                output = model(input_id, attention_mask=mask)
                logits = output.logits

                loss = loss_function(logits.view(-1, logits.size(-1)), eval_label.view(-1))
                    
                eval_loss_arr.append(loss.item())

                predictions = logits.argmax(dim=-1)

                eval_metrics.update(predictions, eval_label)

        train_results = train_metrics.return_avg_metrics()
        eval_results = eval_metrics.return_avg_metrics()
        
        log_dict.update({
            'train/f1_avg': train_results['avg_f1_score'],
            'train/f1_strict': train_results['strict']['f1_score'],
            'train/f1_ent_type': train_results['ent_type']['f1_score'],
            'train/f1_partial': train_results['partial']['f1_score'],
            'train/f1_exact': train_results['exact']['f1_score'],
            'train/loss': sum(train_loss_arr)/len(train_loss_arr)
        })
            
        log_dict.update({
            'eval/f1_avg': eval_results['avg_f1_score'],
            'eval/f1_strict': eval_results['strict']['f1_score'],
            'eval/f1_ent_type': eval_results['ent_type']['f1_score'],
            'eval/f1_partial': eval_results['partial']['f1_score'],
            'eval/f1_exact': eval_results['exact']['f1_score'],
            'eval/loss': sum(eval_loss_arr)/len(eval_loss_arr)
        })
        
        wandb.log(log_dict)

        print(f"Epoch {epoch+1} of {epochs} finished!")
        print(f"TRAIN\nMetrics {train_results}\n")
        print(f"VALIDATION\nMetrics {eval_results}\n")
        print(f"TRAIN\nLoss {sum(train_loss_arr)/len(train_loss_arr)}\n")
        print(f"VALIDATION\nLoss {sum(eval_loss_arr)/len(eval_loss_arr)}\n")
    
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
    
    loss_function = FocalLoss()
    
    test_loss_arr = []

    with torch.no_grad():

        for test_data in test_dataloader:

            test_label = test_data['entity'].to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)
            input_id = test_data['input_ids'].squeeze(1).to(device)

            output = model(input_id, attention_mask=mask)
            logits = output.logits
            
            loss = loss_function(logits.view(-1, logits.size(-1)), test_label.view(-1))
            
            test_loss_arr.append(loss.item())
            
            predictions = logits.argmax(dim=-1)

            test_metrics.update(predictions, test_label)

        test_results = test_metrics.return_avg_metrics()

        print(f"TEST\nMetrics {test_results}\n")
        print(f"TEST\nLoss {sum(test_loss_arr)/len(test_loss_arr)}\n")

    return test_results
