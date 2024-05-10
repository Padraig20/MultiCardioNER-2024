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

args = parser.parse_args()

if args.dataset not in ['es', 'it', 'en']:
    raise ValueError("Dataset must be either es, it or en.")

folder_name = f"../datasets/test+background/{args.dataset}/"
output_file = args.output

import os
import csv
import spacy
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

tokenizer = AutoTokenizer.from_pretrained(f"tok_{args.input}")
model = AutoModelForTokenClassification.from_pretrained(f"model_{args.input}")

model.config.id2label = ids_to_label
model.config.label2id = label_to_ids

ner_model = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities_from_text(text):
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
            start = entity['start'] + start_offset
            end = entity['end'] + start_offset
            
            entities.append((entity_text, entity_type, start, end))

    return entities

def write_entities_to_tsv(filename, text, entities):
    with open(output_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        
        for text, entity_type, start, end in entities:
            if text.strip() == '':
                continue
            writer.writerow([filename, entity_type, start, end, text])

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['filename', 'label', 'start_span', 'end_span', 'text'])

for file_name in tqdm(os.listdir(folder_name)):
    if file_name.endswith(".txt"):
        filename = file_name[:-4]
        with open(os.path.join(folder_name, file_name), 'r') as file:
            content = file.read().replace('\n', ' ')
        extracted_entities = extract_entities_from_text(content)
        write_entities_to_tsv(filename, content, extracted_entities)

print(f"Entities extracted from {len(os.listdir(folder_name))} files in {folder_name} and saved to {output_file}.")