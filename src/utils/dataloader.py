import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.models import get_tokenizer
import re

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
            
            train_filenames = train_data['filename'].unique()
            val_filenames = val_data['filename'].unique()

            train_dataset = Cutoff_Dataset(train_data, train_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train')
            val_dataset = Cutoff_Dataset(val_data, val_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train')
            
            test_data = pd.read_csv(f"../datasets/track1/cardioccc_dev/tsv/multicardioner_track1_cardioccc_dev.tsv", 
               names=['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'], 
               sep="\t", header=0)
            test_data = test_data.drop(columns=['ann_id', 'label', 'text']) #remove unnecessary columns

            test_filenames = test_data['filename'].unique()
        
            test_dataset = Cutoff_Dataset(test_data, test_filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'cardioccc_dev')

            return train_dataset, val_dataset, test_dataset
        else:
            dataset = Cutoff_Dataset(data, filenames, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens, 'distemist_train')
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
            
            _, tokens, _ = self.get_tokenized_file(filename)
            
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
    
    def prepare_input(self, text, labels):

        sen_code = self.tokenizer.encode_plus(text,
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = self.max_tokens, # maximum tokens of a sentence
            padding='max_length',
            return_attention_mask=True, # generates the attention mask
            truncation = True
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
    
    def tokenize_with_positions(self, text):
        
        tokens = self.tokenizer.tokenize(text)
        positions = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        positions = positions[1:len(positions)-1]

        return tokens, positions

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
            text = text.strip().replace("\n", " ")
        
        tokens, token_positions = self.tokenize_with_positions(text)
        
        #print(tokens)
        #print(token_positions)
        
        tokens, labels = self.align_annotations(tokens, annotations, token_positions)
                    
        return text, tokens, labels

    def __getitem__(self, idx):
        """
        Takes the current document with its labels and tokenizes it on-the-fly with the correct format.

        Returns:
        item (torch.tensor): Tensor which can be fed into model.
        """
        
        # Get the filename
        file_idx, start_token_idx, end_token_idx = self.windows[idx]
        filename = self.filenames[file_idx]
        text, tokens, labels = self.get_tokenized_file(filename)
        
        #segment_text = text
        
        segment_tokens = tokens[start_token_idx:end_token_idx]
        segment_labels = labels[start_token_idx:end_token_idx]
        
        item = self.prepare_input(text, segment_labels)

        print("-"*100)
        print("Mask\tEntity\tTokenIDs\tLabels\tTokens")
        for mask, entity, token, label, tok in zip(item['attention_mask'], item['entity'], item['input_ids'], ["-100"] + segment_labels + ["-100"], ["[CLS]"] + segment_tokens + ["[SEP]"]):
            print(f"{mask}\t{entity}\t{token}\t{label}\t{tok}")
        print("-"*100)
        
        #print(item)
        
        return item
    
class Admission_Notes_Dataset(Dataset):
    """ Used for MLM training on admission notes. """
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


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
    
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\n', ' ')

    def parse_annotations(self, annotations_text):
        annotations = []
        for line in annotations_text.splitlines():
            parts = line.strip().split()
            if parts:
                if parts[2].startswith('T'):
                    continue
                tag_type = parts[1]
                start = int(parts[2])
                end = int(parts[3])
                annotations.append((tag_type, start, end))
        return annotations
    
    def apply_annotations_to_tags(self, text, annotations):
        tags = ['O'] * len(text)  # Initialize tags for each character
        for tag_type, start, end in annotations:
            tags[start] = 'B-' + tag_type  # Begin entity
            for i in range(start + 1, end):
                tags[i] = 'I-' + tag_type  # Inside entity
        return tags
    
    def tokenize_text(self, text):
        tokens = []
        positions = []
        for match in re.finditer(r"\w+|\w+(?='s)|'s|['\".,!?;]", text):
            tokens.append(match.group(0))
            positions.append((match.start(), match.end()))
        return tokens, positions
    
    def assign_tags_to_tokens(self, tokens, positions, char_tags):
        token_tags = []
        previous_tag = None
        for token, (start, end) in zip(tokens, positions):
            token_char_tags = char_tags[start:end]
            common_tag = max(set(token_char_tags), key=token_char_tags.count) if token_char_tags else 'O'
            if common_tag == 'O':
                token_tags.append('O')
            else:
                if common_tag != previous_tag:
                    token_tags.append('B-' + common_tag.split('-')[-1])
                else:
                    token_tags.append(common_tag)
            previous_tag = common_tag
        return token_tags
    
    def tokenize_and_preserve_labels(self, sentence, text_labels, max_tokens):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            if(len(tokenized_sentence)>=max_tokens): #truncate
                return tokenized_sentence, labels

            tokenized_sentence.extend(tokenized_word)

            if label.startswith("B-"):
                labels.extend([label])
                labels.extend([self.ids_to_label.get(self.label_to_ids.get(label)+1)]*(n_subwords-1))
            else:
                labels.extend([label] * n_subwords)

        return tokenized_sentence, labels
    
    def tokenize_and_align_labels(self, tokens, labels):
    
        t_sen, t_labl = self.tokenize_and_preserve_labels(tokens, labels, self.max_tokens)

        sen_code = self.tokenizer.encode_plus(tokens,
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = self.max_tokens, # maximum tokens of a sentence
            padding='max_length',
            is_split_into_words=True,
            return_attention_mask=True, # generates the attention mask
            truncation = True
            )

        #shift labels (due to [CLS] and [SEP])
        labels = [-100]*self.max_tokens #-100 is ignore token
        for i, tok in enumerate(t_labl):
            if tok != None and i < self.max_tokens-1:
                labels[i+1]=self.label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        
        item['entity'] = torch.as_tensor(labels)
    
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
                
        text = self.read_file(f"../datasets/track1/{self.folder_lbl}/brat/{filename}.txt")
        annotations_text = self.read_file(f"../datasets/track1/{self.folder_lbl}/brat/{filename}.ann")
        annotations = self.parse_annotations(annotations_text)
        char_tags = self.apply_annotations_to_tags(text, annotations)
        tokens, positions = self.tokenize_text(text)
        token_tags = self.assign_tags_to_tokens(tokens, positions, char_tags)
        
        tokens, labels = tokens, token_tags
        
        item = self.tokenize_and_align_labels(tokens, labels)
        
        return item

    def __getitem__(self, idx):
        """
        Takes the current document with its labels and tokenizes it on-the-fly with the correct format.

        Returns:
        item (torch.tensor): Tensor which can be fed into model.
        """
        
        #  Get the filename
        filename = self.filenames[idx]
        
        # Retrieve and tokenize the file
        item = self.get_tokenized_file(filename)
        
        return item