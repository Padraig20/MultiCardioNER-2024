from transformers import BertForTokenClassification, BertForMaskedLM, AutoTokenizer
import torch.nn as nn

model_type = "PlanTL-GOB-ES/bsc-bio-ehr-es"
#model_type = "bert-base-multilingual-cased"

class BertNER(nn.Module):
    """
    Architecture for Named Entity Recognition.
    """
    def __init__(self, tokens_dim):
        super(BertNER,self).__init__()
        self.pretrained = BertForTokenClassification.from_pretrained(model_type, num_labels = tokens_dim)
        #self.pretrained = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels = tokens_dim)


    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out

class BertMLM(nn.Module):
    """
    Architecture for Masked Language Modeling.
    """
    def __init__(self):
        super(BertMLM,self).__init__()
        self.pretrained = BertForMaskedLM.from_pretrained(model_type)

    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out

def get_tokenizer():
    """
    Get tokenizer for BERT.
    """
    return AutoTokenizer.from_pretrained(model_type)