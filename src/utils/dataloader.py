import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Dataloader():
    """
    Dataloader used for loading the dataset used in this project. Also provides a framework for automatic
    tokenization of the data.
    """

    def __init__(self, label_to_ids, ids_to_label, max_tokens):
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens

    def load_dataset(self, full = False):

        data = pd.read_csv(f"../datasets/track1/cardioccc_dev/tsv/multicardioner_track1_cardioccc_dev.tsv", 
               names=['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'], 
               sep="\t", header=0)
        data = data.drop(columns=['ann_id', 'label', 'text']) #remove unnecessary columns

        filenames = data['filename'].unique()
        
        tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer.add_tokens(['B-ENFERMEDAD', 'I-ENFERMEDAD', 'O'])

        if not full:
            #70-10-20 split

            train_data = data.sample(frac=0.7, random_state=7).reset_index(drop=True)

            remaining_data = data.drop(train_data.index).reset_index(drop=True)
            val_data = remaining_data.sample(frac=0.2857, random_state=7).reset_index(drop=True)

            test_data = remaining_data.drop(val_data.index).reset_index(drop=True)

            train_dataset = Custom_Dataset(train_data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            val_dataset = Custom_Dataset(val_data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            test_dataset = Custom_Dataset(test_data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)

            return train_dataset, val_dataset, test_dataset
        else:
            dataset = Custom_Dataset(data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            return dataset

class Custom_Dataset(Dataset):
    """
    Dataset used for loading and tokenizing sentences on-the-fly.
    """

    def __init__(self, data, filenames, tokenizer, label_to_ids, ids_to_label, max_tokens):
        self.data = data
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.stack = []

    def __len__(self):
        return len(self.filenames)
    
    def prepare_input(self, tokens, labels):

        sen_code = self.tokenizer.encode_plus(tokens,
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = self.max_tokens, # maximum tokens of a sentence
            padding='max_length',
            return_attention_mask=True, # generates the attention mask
            truncation = True,
            is_split_into_words=True
            )

        #shift labels (due to [CLS] and [SEP])
        lbls = [-100]*self.max_tokens #-100 is ignore token
        for i, tok in enumerate(labels):
            if tok != None and i < self.max_tokens-1:
                lbls[i+1]=self.label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['entity'] = torch.as_tensor(lbls)

        return item

    def get_tokenized_file(self, filename):
        """
        Tokenizes a file and returns the tokens and labels.

        Args:
        filename (str): Name of the file to tokenize.

        Returns:
        tokens (list): List of tokens.
        labels (list): List of labels.
        """
        
        # get all entities in the document
        entity_documents = self.data[self.data['filename'] == filename]
        
        # extract annotations
        annotations = []
        for _, row in entity_documents.iterrows():
            start_span = row['start_span']
            end_span = row['end_span']
            annotations.append(('ENFERMEDAD', start_span, end_span))
        
        # Load the text
        with open(f"../datasets/track1/cardioccc_dev/brat/{filename}.txt", 'r') as file:
            text = file.read()
        
        tokens = self.tokenizer.tokenize(text)
        token_positions = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        token_positions = token_positions[1:len(token_positions)-1]

        # Align BRAT annotations to tokenized text
        labels = ['O'] * len(tokens)

        for ann_type, start, end in annotations:
            for idx, (tok_start, tok_end) in enumerate(token_positions):
                if tok_start >= start and tok_end <= end:
                    prefix = 'B' if tok_start == start else 'I'
                    labels[idx] = f"{prefix}-{ann_type}"
                    
        return tokens, labels

    def __getitem__(self, idx):
        """
        Takes the current document with its labels and tokenizes it on-the-fly with the correct format.

        Returns:
        item (torch.tensor): Tensor which can be fed into model.
        """
        
        #  Get the filename
        filename = self.filenames[idx]
        
        # Retrieve and tokenize the file
        tokens, labels = self.get_tokenized_file(filename)
          
        # Prepare the input for BERT
        item = self.prepare_input(tokens, labels)
        
        return item