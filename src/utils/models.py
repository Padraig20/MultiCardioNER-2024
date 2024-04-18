import transformers
from transformers import BertForTokenClassification
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F

class BertNER(nn.Module):
    """
    Architecture using neuralmind/bert-base-portuguese-cased.
    """
    def __init__(self, tokens_dim):
        super(BertNER,self).__init__()
        self.pretrained = BertForTokenClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", num_labels = tokens_dim)

    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out