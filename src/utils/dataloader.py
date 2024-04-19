import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.models import get_tokenizer

class Dataloader():
    """
    Dataloader used for loading the dataset used in this project. Also provides a framework for automatic
    tokenization of the data.
    """

    def __init__(self, label_to_ids, ids_to_label, max_tokens, window_stride):
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.window_stride = window_stride

    def load_dataset(self, full = False):

        data = pd.read_csv(f"../datasets/track1/distemist_train/tsv/multicardioner_track1_distemist_train.tsv", 
               names=['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'], 
               sep="\t", header=0)
        data = data.drop(columns=['ann_id', 'label', 'text']) #remove unnecessary columns

        filenames = data['filename'].unique()
        
        tokenizer = get_tokenizer()

        if not full:
            #80-20 split

            train_data = data.sample(frac=0.8, random_state=7).reset_index(drop=True)
            val_data = data.drop(train_data.index).reset_index(drop=True)
            
            train_filenames = train_data['filename'].unique()[:20] #TODO
            val_filenames = val_data['filename'].unique()[:5] #TODO

            train_dataset = Sliding_Window_Dataset(train_data, train_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train', self.window_stride)
            val_dataset = Sliding_Window_Dataset(val_data, val_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train', self.window_stride)
            
            test_data = pd.read_csv(f"../datasets/track1/cardioccc_dev/tsv/multicardioner_track1_cardioccc_dev.tsv", 
               names=['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'], 
               sep="\t", header=0)
            test_data = test_data.drop(columns=['ann_id', 'label', 'text']) #remove unnecessary columns

            test_filenames = test_data['filename'].unique()[:5] #TODO
        
            test_dataset = Sliding_Window_Dataset(test_data, test_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'cardioccc_dev', self.window_stride)

            return train_dataset, val_dataset, test_dataset
        else:
            dataset = Sliding_Window_Dataset(data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train', self.window_stride)
            return dataset

class Sliding_Window_Dataset(Dataset):
    """
    Dataset used for loading and tokenizing sentences on-the-fly with sliding windows to ensure increased data capture.
    """

    def __init__(self, data, filenames, tokenizer, label_to_ids, ids_to_label, max_tokens, folder_lbl, overlap):
        self.data = data
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.folder_lbl = folder_lbl
        self.overlap = overlap
        self.windows = self._create_windows()

    def __len__(self):
        return len(self.windows)
    
    def _create_windows(self):
        windows = []
        for idx, filename in enumerate(self.filenames):
            
            tokens, _ = self.get_tokenized_file(filename)
            
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            total_tokens = len(token_ids)
                        
            start_token_idx = 0
            window_size = self.max_tokens - 2 # -2 for [CLS] and [SEP]
            step_size = window_size - self.overlap
        
            while start_token_idx < total_tokens:
                end_token_idx = start_token_idx + window_size
                end_token_idx = min(end_token_idx, total_tokens)
                windows.append((idx, start_token_idx, end_token_idx)) # don't save file to save memory
                start_token_idx += step_size

        return windows
    
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
    
    def align_annotations(self, tokens, annotations, token_positions):
        labels = ['O'] * len(tokens)

        for ann_type, start, end in annotations:
            for idx, (tok_start, tok_end) in enumerate(token_positions):
                if tok_start >= start and tok_end <= end and idx < len(labels):
                    prefix = 'B' if tok_start == start else 'I'
                    labels[idx] = f"{prefix}-{ann_type}"
        
        return tokens, labels

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
        with open(f"../datasets/track1/{self.folder_lbl}/brat/{filename}.txt", 'r') as file:
            text = file.read()
        
        tokens = self.tokenizer.tokenize(text)
        token_positions = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        token_positions = token_positions[1:len(token_positions)-1]
                    
        return self.align_annotations(tokens, annotations, token_positions)

    def __getitem__(self, idx):
        """
        Takes the current document with its labels and tokenizes it on-the-fly with the correct format.

        Returns:
        item (torch.tensor): Tensor which can be fed into model.
        """
        
        # Get the filename
        file_idx, start_token_idx, end_token_idx = self.windows[idx]
        filename = self.filenames[file_idx]
        tokens, labels = self.get_tokenized_file(filename)
        
        segment_tokens = tokens[start_token_idx:end_token_idx]
        segment_labels = labels[start_token_idx:end_token_idx]
        
        item = self.prepare_input(segment_tokens, segment_labels)
                
        # attention mask is not correct, pads should be set to 0 for no attention - 0 for [CLS] and 2 for [SEP] in Spanish BERT
        #item['attention_mask'] = torch.as_tensor([0 if token != 101 and token != 102 and entity == -100 else 1 for token, entity in zip(item['input_ids'], item['entity'])])
        item['attention_mask'] = torch.as_tensor([0 if token != 0 and token != 2 and entity == -100 else 1 for token, entity in zip(item['input_ids'], item['entity'])])
        
        # rectify input_ids - add 1 as [PAD] and rectify 2 as [SEP] before [PAD]
        item['input_ids'] = torch.as_tensor([1 if token != 0 and ent == -100 else token for token, ent in zip(item['input_ids'], item['entity'])])

        new_ids = []
        for i in range(0, len(item['input_ids'])):
            if i > 0 and item['entity'][i] == -100 and item['entity'][i-1] != -100:
                new_ids.append(2)
            else:
                new_ids.append(item['input_ids'][i])
        
        item['input_ids'] = torch.as_tensor(new_ids)

        #print("-"*100)
        #print("Mask\tEntity\tTokenIDs\tLabels\tTokens")
        #for mask, entity, token, label, tok in zip(item['attention_mask'], item['entity'], item['input_ids'], ["-100"] + segment_labels + ["-100"], ["[CLS]"] + segment_tokens + ["[SEP]"]):
        #    print(f"{mask}\t{entity}\t{token}\t{label}\t{tok}")
        #print("-"*100)
        
        #print(item)
        
        return item
    

class Cutoff_Dataset(Dataset):
    """
    Dataset used for loading and tokenizing sentences on-the-fly. Excess data is simply cutoff.
    """

    def __init__(self, data, filenames, tokenizer, label_to_ids, ids_to_label, max_tokens, folder_lbl):
        self.data = data
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.folder_lbl = folder_lbl

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
        with open(f"../datasets/track1/{self.folder_lbl}/brat/{filename}.txt", 'r') as file:
            text = file.read()
        
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        token_positions = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        token_positions = token_positions[1:len(token_positions)-1]

        # Align BRAT annotations to tokenized text
        labels = ['O'] * len(tokens)

        for ann_type, start, end in annotations:
            for idx, (tok_start, tok_end) in enumerate(token_positions):
                if tok_start >= start and tok_end <= end and idx < len(labels):
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