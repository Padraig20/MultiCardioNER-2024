import argparse

parser = argparse.ArgumentParser(
        description='This class is used to pre-train a transformer-based model on admission notes.')

parser.add_argument('-o', '--output', type=str, required=True,
                    help='Choose where the output tsv file should be saved.')
parser.add_argument('-in', '--input', type=str, required=True,
                    help='Import your transformer-based model and tokenizer.')
parser.add_argument('-d', '--dataset', type=str, default="es",
                    help='Choose the dataset you want to evaluate the model on. Choose from: es, it, en')
parser.add_argument('-t', '--type', type=str, default="ENFERMEDAD",
                    help='Choose the entity type. Choose from: ENFERMEDAD, FARMACO.')
parser.add_argument('-ckp', '--checkpoint', type=str, default=None,
                    help='Choose the checkpoint output you want to evaluate the model on. Must be a .csv file. Useful if scrpit terminated during evaluation.')
parser.add_argument('-sp', '--special_model', type=str, default=None,
                    help='Choose whether you used a special model, i.e. special tokenization and labels. Choose from: lcampillos/roberta-es-clinical-trials-ner, biobit, biobert')
parser.add_argument('-test', '--test_set', type=bool, default=False,
                    help='Choose whether you want to evaluate the model on the test set. If False, the model will be evaluated on both the background set and the test set.')
parser.add_argument('-dev', '--dev_set', type=bool, default=False,
                    help='Choose whether you want to evaluate the model on the dev set. If False, the model will be evaluated on teh test set.')
parser.add_argument('-strat', '--strategy', type=str, default='sentences',
                    help='Choose the strategy you want to use to extract entities. Choose from: cutoff, sentences, chunks. Default is sentences.')

args = parser.parse_args()

if args.dataset not in ['es', 'it', 'en']:
    raise ValueError("Dataset must be either es, it or en.")

if args.special_model and args.special_model not in ['lcampillos/roberta-es-clinical-trials-ner', 'biobit', 'biobert']:
    raise ValueError("Special model must be either lcampillos/roberta-es-clinical-trials-ner, biobert or biobit.")

if args.test_set and args.dev_set:
    raise ValueError("Either test_set or dev_set must be set to True.")

if args.strategy not in ['cutoff', 'sentences', 'chunks']:
    raise ValueError("Strategy must be either cutoff, sentences or chunks.")

if args.dev_set:
    if args.type == 'ENFERMEDAD':
        folder_name = "../datasets/track1/cardioccc_dev/brat/"
    else:
        folder_name = f"../datasets/track2/cardioccc_dev/{args.dataset}/brat/"
else:
    folder_name = f"../datasets/test+background/{args.dataset}/"

output_file = args.output

print(f"Extracting entities from {folder_name} and saving to {output_file}.")

import os
import csv
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

if args.dataset == 'es':
    nlp = spacy.load("es_core_news_sm")
elif args.dataset == 'it':
    nlp = spacy.load("it_core_news_sm")
else:
    nlp = spacy.load("en_core_web_sm")

label_to_ids = {
    f'B-{args.type}': 0,
    f'I-{args.type}': 1,
    'O': 2
}

ids_to_label = {
    0:f'B-{args.type}',
    1:f'I-{args.type}',
    2:'O'
}

if args.special_model:
    if args.special_model == "lcampillos/roberta-es-clinical-trials-ner":
    
        if args.type == 'ENFERMEDAD':
            label_to_ids = {
                'B-ANAT': 0,
                'B-CHEM': 2,
                'B-ENFERMEDAD': 4, #DISO
                'B-PROC': 6,
                'I-ANAT': 1,
                'I-CHEM': 3,
                'I-ENFERMEDAD': 5, #DISO
                'I-PROC': 7,
                'O': 8
            }

            ids_to_label = {
                0:'O',
                1:'O',
                2:'O',
                3:'O',
                4:'B-ENFERMEDAD', #DISO
                5:'I-ENFERMEDAD', #DISO
                6:'O',
                7:'O',
                8:'O'
            }
        else:
            label_to_ids = {
                'B-ANAT': 0,
                'B-FARMACO': 2, #CHEM
                'B-DISO': 4,
                'B-PROC': 6,
                'I-ANAT': 1,
                'I-FARMACO': 3, #CHEM
                'I-DISO': 5,
                'I-PROC': 7,
                'O': 8
            }

            ids_to_label = {
                0:'O',
                1:'O',
                2:'B-FARMACO', #CHEM
                3:'I-FARMACO', #CHEM
                4:'O',
                5:'O',
                6:'O',
                7:'O',
                8:'O'
            }

tokenizer = AutoTokenizer.from_pretrained(f"tok_{args.input}")
model = AutoModelForTokenClassification.from_pretrained(f"model_{args.input}")

max_length = tokenizer.model_max_length

model.config.id2label = ids_to_label
model.config.label2id = label_to_ids

ner_model = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    sentence_offsets = [sent.start_char for sent in doc.sents]

    sentence_entities_batch = ner_model(sentences, batch_size=len(sentences)) #adjust batch size

    entities = []
    for sent_idx, sentence_entities in enumerate(sentence_entities_batch):
        start_offset = sentence_offsets[sent_idx]
        
        for entity in sentence_entities:
            entity_text = entity['word']
            entity_type = entity['entity_group']
            start = entity['start'] + 1 + start_offset # add 1 (for some reason? only if roberta)
            if args.special_model == "lcampillos/roberta-es-clinical-trials-ner" or args.special_model == 'biobit' or args.special_model == 'biobert':
                start -= 1
            end = entity['end'] + start_offset
            
            entities.append((entity_text, entity_type, start, end))

    return entities

def extract_entities_cutoff(text):
    entities_batch = ner_model([text], batch_size=1)

    entities = []
    for entity in entities_batch[0]:
        entity_text = entity['word']
        entity_type = entity['entity_group']
        start = entity['start'] + 1
        if args.special_model in ["lcampillos/roberta-es-clinical-trials-ner", "biobit", "biobert"]:
            start -= 1
        end = entity['end']
    
        entities.append((entity_text, entity_type, start, end))
    
    return entities

def chunk_text(text):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(tokenizer.tokenize(word))
        if current_length + word_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_entities_chunks(text):
    chunks = chunk_text(text)
    entities = []
    current_offset = 0

    for chunk in chunks:
        chunk_entities = ner_model([chunk], batch_size=1)[0]
        for entity in chunk_entities:
            entity_text = entity['word']
            entity_type = entity['entity_group']
            start = entity['start'] + current_offset
            end = entity['end'] + current_offset

            if args.special_model in ["lcampillos/roberta-es-clinical-trials-ner", "biobit", "biobert"]:
                start -= 1

            entities.append((entity_text, entity_type, start, end))
        
        current_offset += len(chunk)

    return entities

def extract_entities_from_text(text):
    
    if args.strategy == 'cutoff':
        entities = extract_entities_cutoff(text)
    elif args.strategy == 'sentences':
        entities = extract_entities_sentences(text)
    elif args.strategy == 'chunks':
        entities = extract_entities_chunks(text)

    # roberta based model requires more reconstruction
    if args.special_model == "lcampillos/roberta-es-clinical-trials-ner":
        reconstructed_entities = []
        for entity in entities:
            entity_text = entity[0]
            entity_type = entity[1]
            start = entity[2]
            end = entity[3]
        
            if entity_text.startswith(' '):
                entity_text = entity_text[1:]
            else:
                if reconstructed_entities:
                    prev_entity = reconstructed_entities[-1]
                    prev_entity_text = prev_entity[0]
                    prev_entity_type = prev_entity[1]
                    prev_start = prev_entity[2]
                    prev_end = prev_entity[3]
                
                    if prev_end == start:
                        reconstructed_text = prev_entity_text + entity_text
                        reconstructed_entities[-1] = (reconstructed_text, prev_entity_type, prev_start, end)
                        continue
        
            reconstructed_entities.append((entity_text, entity_type, start, end))

        entities = reconstructed_entities
    
    # biobit leaves subwords for some reason
    elif args.special_model == "biobit" or args.special_model == "biobert":
        reconstructed_entities = []
        for entity in entities:
            entity_text = entity[0]
            entity_type = entity[1]
            start = entity[2]
            end = entity[3]
                
            if not entity_text.startswith('##'):
                reconstructed_entities.append((entity_text, entity_type, start, end))
            
        entities = reconstructed_entities
    
    return entities

def write_entities_to_tsv(filename, text, entities):
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        
        for text, entity_type, start, end in entities:
            try:
                if text.strip() == '':
                    continue
                writer.writerow([filename, entity_type, start, end, text])
            except Exception as e:
                print(f"Error writing entity to TSV: {e}")
                print(filename, entity_type, start, end, text)

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['filename', 'label', 'start_span', 'end_span', 'text'])

def load_unique_filenames(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', usecols=[0], header=None)
    unique_filenames = df[0].unique().tolist()
    return unique_filenames

def load_testset_filenames(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        filenames = file.read().splitlines()
    return filenames

if args.checkpoint:
    filenames = load_unique_filenames(args.checkpoint)
    
if args.test_set:
    filenames = load_testset_filenames(f"../datasets/multicardioner_test_fnames.txt")

if not args.test_set:
    for file_name in tqdm(os.listdir(folder_name)):
        if file_name.endswith(".txt"):
            filename = file_name[:-4]
            if args.checkpoint:
                if filename in filenames:
                    print(f"Skipping {filename} as it is already in the checkpoint.")
                    continue
            with open(os.path.join(folder_name, file_name), 'r', encoding='utf-8') as file:
                content = file.read().replace('\n', ' ')
            extracted_entities = extract_entities_from_text(content)
            write_entities_to_tsv(filename, content, extracted_entities)
else:
    for file_name in tqdm(filenames):
        with open(os.path.join(folder_name, file_name + ".txt"), 'r', encoding='utf-8') as file:
            content = file.read().replace('\n', ' ')
        extracted_entities = extract_entities_from_text(content)
        write_entities_to_tsv(file_name, content, extracted_entities)

print(f"Entities extracted from {len(os.listdir(folder_name))} files in {folder_name} and saved to {output_file}.")