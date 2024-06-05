import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_metric
from utils.metric_tracking import MetricsTracking
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        
        if torch.cuda.is_available() and self.alpha is not None:
            self.alpha = self.alpha.to(inputs.device)
        
        # Calculate Log Softmax
        log_softmax = F.log_softmax(inputs, dim=1)
        
        #print(targets.unsqueeze(1))
        # Gather the log probabilities by the actual class indices
        targets = torch.where(targets == -100, torch.tensor(0, device=targets.device), targets)
        gather = torch.gather(log_softmax, 1, targets.unsqueeze(1))

        
        # Calculate the focal loss
        focal_loss = -(1 - torch.exp(gather)) ** self.gamma * gather

        # Apply the alpha weighting
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)  # This gathers the correct alpha per target
            at = at.unsqueeze(1)  # Ensure it has the shape [batch_size, 1] if not already
            focal_loss *= at
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss

class MetricsForHuggingfaceTrainer:
    def __init__(self, label_list, entity_type):
        self.metric = load_metric("seqeval")
        self.label_list = label_list
        self.entity_type = entity_type
    
    def conll(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        flat_true_predictions = [p for sublist in true_predictions for p in sublist]

        flat_true_labels = [l for sublist in true_labels for l in sublist]

        tracker = MetricsTracking(self.entity_type, tensor_input=False)
        tracker.update(flat_true_predictions, flat_true_labels)

        return tracker.return_avg_metrics()
    
    def seqeval(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        for k in results.keys():
            if (k not in flattened_results.keys()):
                flattened_results[k+"_f1"]=results[k]["f1"]

        return flattened_results
    
    def get_metric(self, metric_name):
        if metric_name == "conll":
            return self.conll
        elif metric_name == "seqeval":
            return self.seqeval
        else:
            raise ValueError("Invalid metric name")