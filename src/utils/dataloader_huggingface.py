from datasets import Dataset
import pandas as pd
import torch
import spacy
from tqdm import tqdm

def load_ner_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []
    
    for line in lines:
        line = line.strip()
        if line == "":
            if current_tokens and current_labels:
                tokens.append(current_tokens)
                labels.append(current_labels)
                current_tokens = []
                current_labels = []
        else:
            parts = line.split()
            token = parts[0]
            label = parts[1]
            current_tokens.append(token)
            current_labels.append(label)
    
    if current_tokens and current_labels:
        tokens.append(current_tokens)
        labels.append(current_labels)
    
    df = pd.DataFrame({"tokens": tokens, "labels": labels})
    
    return Dataset.from_pandas(df)

######################################################
#########         CutoffLengthDataset         ########
######################################################

class CutoffLengthDataset():
    def __init__(self, max_length, tokenizer, ids_to_label, label_to_ids):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ids_to_label = ids_to_label
        self.label_to_ids = label_to_ids

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            if(len(tokenized_sentence)>=self.max_length): #truncate
                return tokenized_sentence, labels

            tokenized_sentence.extend(tokenized_word)

            if label.startswith("B-"):
                labels.extend([label])
                labels.extend([self.ids_to_label.get(self.label_to_ids.get(label)+1)]*(n_subwords-1))
            else:
                labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def tokenize_and_align_labels(self, examples):
    
        t_sen, t_labl = self.tokenize_and_preserve_labels(examples['tokens'], examples['labels'])

        sen_code = self.tokenizer.encode_plus(examples['tokens'],
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = self.max_length, # maximum tokens of a sentence
            padding='max_length',
            is_split_into_words=True,
            return_attention_mask=True, # generates the attention mask
            truncation = True
            )

        #shift labels (due to [CLS] and [SEP])
        labels = [-100]*self.max_length #-100 is ignore token
        for i, tok in enumerate(t_labl):
            if tok != None and i < self.max_length-1:
                labels[i+1]=self.label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        
        item['entity'] = torch.as_tensor(labels)
    
        return item
    
    def prepare_dataset(self, dataset):
        tokenized_datasets = dataset.map(self.tokenize_and_align_labels)

        tokenized_datasets = tokenized_datasets.remove_columns('tokens')
        tokenized_datasets = tokenized_datasets.remove_columns('labels')
        tokenized_datasets = tokenized_datasets.rename_column("entity", "labels")

        return tokenized_datasets
    
    def get_dataset(self, file_path):
        dataset = load_ner_dataset(file_path)
        return self.prepare_dataset(dataset)
    
######################################################
########         SlidingWindowDataset         ########
######################################################

class SlidingWindowDataset():
    def __init__(self, max_length, tokenizer, ids_to_label, label_to_ids, stride):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ids_to_label = ids_to_label
        self.label_to_ids = label_to_ids
        self.stride = stride
        
    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            if(len(tokenized_sentence)>=self.max_length): #truncate
                return tokenized_sentence, labels

            tokenized_sentence.extend(tokenized_word)

            if label.startswith("B-"):
                labels.extend([label])
                labels.extend([self.ids_to_label.get(self.label_to_ids.get(label)+1)]*(n_subwords-1))
            else:
                labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def sliding_window_tokenize_and_align_labels(self, sentence_tokens, sentence_labels):
        tokenized_sentences = []
        label_sentences = []
    
        start = 0
        while start < len(sentence_tokens):
            end = start + self.max_length
            window_tokens = sentence_tokens[start:end]
            window_labels = sentence_labels[start:end]
        
            # tokenize and preserve labels for this window
            tokens, labels = self.tokenize_and_preserve_labels(window_tokens, window_labels)
        
            encoding = self.tokenizer.encode_plus(sentence_tokens,
                add_special_tokens=True, # adds [CLS] and [SEP]
                max_length = self.max_length, # maximum tokens of a sentence
                padding='max_length',
                is_split_into_words=True,
                return_attention_mask=True, # generates the attention mask
                truncation = True
            )
        
            # shift due to special tokens
            window_labels = [-100] + labels[:self.max_length - 2] + [-100]  # Pad for [CLS] and [SEP]
            window_labels = window_labels[:self.max_length]  # label length matches encoding length

            tokenized_sentences.append(encoding)
            label_sentences.append(window_labels)

            start += self.stride
    
        return tokenized_sentences, label_sentences

    def prepare_dataset(self, dataset):
        tokenized_data = []
        for tokens, labels in zip(dataset['tokens'], dataset['labels']):
            tokenized, labeled = self.sliding_window_tokenize_and_align_labels(tokens, labels)
            
            tokenized_data.extend([{"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": [self.label_to_ids[label] if label != -100 else -100 for label in labels]} for tokens, labels in zip(tokenized, labeled)])
        
        dataset_pd = pd.DataFrame(tokenized_data)
        
        tokenized_datasets = Dataset.from_dict(dataset_pd)

        return tokenized_datasets
    
    def get_dataset(self, file_path):
        dataset = load_ner_dataset(file_path)
        return self.prepare_dataset(dataset)

######################################################
########         SingleSentenceDataset         #######
######################################################

class SingleSentenceDataset():
    def __init__(self, max_length, tokenizer, ids_to_label, label_to_ids, language):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ids_to_label = ids_to_label
        self.label_to_ids = label_to_ids
        self.language = language
        self.nlp = spacy.load(f"{self.language}_core_web_sm")
        
    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            if(len(tokenized_sentence)>=self.max_length): #truncate
                return tokenized_sentence, labels

            tokenized_sentence.extend(tokenized_word)

            if label.startswith("B-"):
                labels.extend([label])
                labels.extend([self.ids_to_label.get(self.label_to_ids.get(label)+1)]*(n_subwords-1))
            else:
                labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def split_into_sentences(self, tokens, labels):
        text = " ".join(tokens)

        doc = self.nlp(text)

        sentences = [sent.text for sent in doc.sents]

        sentence_tokens = []
        sentence_labels = []

        current_position = 0

        for sentence in sentences:
            sent_tokens = sentence.split()
            sent_length = len(sent_tokens)
    
            sentence_tokens.append(tokens[current_position:current_position + sent_length])
            sentence_labels.append(labels[current_position:current_position + sent_length])
    
            current_position += sent_length
        
        return sentence_tokens, sentence_labels

    def tokenize_and_align_labels(self, tokens, labels):
        tokenized_sentences = []
        label_sentences = []

        sentences_tokens, sentences_labels = self.split_into_sentences(tokens, labels)
        
        for sentence_tokens, sentence_labels in zip(sentences_tokens, sentences_labels):
            _, tokenized_labels = self.tokenize_and_preserve_labels(sentence_tokens, sentence_labels)
        
            encoding = self.tokenizer.encode_plus(sentence_tokens,
                add_special_tokens=True, # adds [CLS] and [SEP]
                max_length = self.max_length, # maximum tokens of a sentence
                padding='max_length',
                is_split_into_words=True,
                return_attention_mask=True, # generates the attention mask
                truncation = True
            )
        
            # shift due to special tokens
            tokenized_labels = [-100] + tokenized_labels[:self.max_length - 2] + [-100]
            tokenized_labels = tokenized_labels[:self.max_length]  # label length matches encoding length
            
            tokenized_sentences.append(encoding)
            label_sentences.append(tokenized_labels)
    
        return tokenized_sentences, label_sentences

    def prepare_dataset(self, dataset):
        tokenized_data = []
        print("Tokenizing and aligning labels for single sentences. This might take a while...")
        for tokens, labels in tqdm(zip(dataset['tokens'], dataset['labels'])):
            tokenized, labeled = self.tokenize_and_align_labels(tokens, labels)
            
            tokenized_data.extend([{"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": [self.label_to_ids[label] if label != -100 else -100 for label in labels]} for tokens, labels in zip(tokenized, labeled)])
        
        dataset_pd = pd.DataFrame(tokenized_data)
        
        tokenized_datasets = Dataset.from_dict(dataset_pd)

        return tokenized_datasets
    
    def get_dataset(self, file_path):
        dataset = load_ner_dataset(file_path)
        return self.prepare_dataset(dataset)