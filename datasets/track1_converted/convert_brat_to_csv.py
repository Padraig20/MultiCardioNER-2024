import re
import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse_annotations(annotations_text):
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

def apply_annotations_to_tags(text, annotations):
    tags = ['O'] * len(text)  # Initialize tags for each character
    for tag_type, start, end in annotations:
        tags[start] = 'B-' + tag_type  # Begin entity
        for i in range(start + 1, end):
            tags[i] = 'I-' + tag_type  # Inside entity
    return tags

def tokenize_text(text):
    tokens = []
    positions = []
    for match in re.finditer(r"\w+|\w+(?='s)|'s|['\".,!?;]", text):
        tokens.append(match.group(0))
        positions.append((match.start(), match.end()))
    return tokens, positions

def assign_tags_to_tokens(tokens, positions, char_tags):
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

def write_to_csv(filename, text, token_tags):
    with open(filename, 'a') as file:
        text = text.replace('\n', ' ')
        token_tags = ' '.join(token_tags)
        file.write(f"{text}|{token_tags}\n")

def convert_brat_to_csv(text_file, ann_file, output_csv):
    text = read_file(text_file).replace('\n', ' ')
    annotations_text = read_file(ann_file)
    annotations = parse_annotations(annotations_text)
    char_tags = apply_annotations_to_tags(text, annotations)
    tokens, positions = tokenize_text(text)
    token_tags = assign_tags_to_tokens(tokens, positions, char_tags)
    print(annotations_text)
    print(token_tags)
    #print(char_tags)
    #print(token_tags)
    write_to_csv(output_csv, text, token_tags)


def read_files_from_directory(directory):
    files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.ann'):
            file_path = os.path.join(directory, file_name)
            files.append(str(file_path)[:len(file_path)-4])
    return files

directory = '../track1/cardioccc_test/brat/'
files = read_files_from_directory(directory)
unique_files = set(files)

for file in files:
    text_file = file + ".txt"
    ann_file = file + ".ann"
    output_csv = "./test/all_test.csv"
    #print(file)
    convert_brat_to_csv(text_file, ann_file, output_csv)