import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def describe_cvs(path):
    data = pd.read_csv(path, names=['text', 'entity'], header=None, sep="|")
    print(f"Data loaded into dataframe:\n\n{data.head(10)}\n\n")

    unique_tags = data['entity'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    print(f"Entities in the data:\n\n{unique_tags}\n\n")

    all_tokens = data['entity'].apply(lambda x: len(x.split(" "))).sum(axis = 0)
    print(f"All tokens in file: {all_tokens}")

    #sent_len = data['text'].apply(len)
    sent_len = data['entity'].apply(lambda x: len(x.split(" ")))
    longest_sentence = sent_len.max()
    shortest_sentence = sent_len.min()
    median_sentence = sent_len.median()
    mean_sentece = sent_len.mean()
    print("Patient Note Length Statistics:\n")
    print(f"min: {shortest_sentence}")
    print(f"max: {longest_sentence}")
    print(f"median: {median_sentence}")
    print(f"mean: {mean_sentece}")

    plt.figure(figsize=(10, 6))
    sent_len.plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Patient Note Lengths ({ent_type}, {lang}, {dataset})')
    plt.xlabel('Length of Patient Note')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    sns.set(style="whitegrid")

    palette = sns.color_palette("coolwarm", 7)

    plt.figure(figsize=(15, 6))
    sns.boxplot(sent_len, color=palette[3], saturation=0.75, orient="h")

    plt.title(f'Boxplot of Patient Note Lengths ({ent_type}, {lang}, {dataset})', fontsize=16)
    plt.xlabel('Sentence Length', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks([])

    plt.show()

def plot_relative_positions(relative_positions):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(relative_positions, bw_adjust=0.5, fill=True)
    plt.xlabel("Relative Word Position")
    plt.ylabel("Density")
    plt.title(f"Entities Across Relative Word Positions ({ent_type}, {lang}, {dataset})")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(relative_positions, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Relative Word Position")
    plt.ylabel("Frequency")
    plt.title(f"Entities Across Relative Word Positions ({ent_type}, {lang}, {dataset})")
    plt.grid(axis='y')
    plt.show()

    bin_edges = np.linspace(0, 1, 21)  # 20 bins from 0% to 100%
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(relative_positions, bins=bin_edges)

    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, hist, marker='o', linestyle='-', color='purple')
    plt.xlabel("Relative Word Position")
    plt.ylabel("Frequency")
    plt.title(f"Entities Across Relative Word Positions ({ent_type}, {lang}, {dataset})")
    plt.grid(True)
    plt.show()
    
def extract_entities(annotations):
    return [(i, tag) for i, tag in enumerate(annotations) if tag in [f'B-{ent_type}', f'I-{ent_type}']]

def calculate_relative_positions(dataframe):
    relative_positions = []
    for _, row in dataframe.iterrows():
        annotations = row['Annotations']
        positions = [i for i, tag in enumerate(annotations) if tag in [f'B-{ent_type}', f'I-{ent_type}']]
        total_length = len(annotations)
        relative_positions.extend([pos / total_length for pos in positions])
    return relative_positions
    
def prepare_dataset(path):
    dataframe = pd.read_csv(path, sep='|', header=None, names=['Text', 'Annotations'])
    dataframe['Annotations'] = dataframe['Annotations'].str.strip().str.split(' ')
    dataframe['EntityPositions'] = dataframe['Annotations'].apply(extract_entities)
    relative_positions = calculate_relative_positions(dataframe)
    return relative_positions

import argparse

parser = argparse.ArgumentParser(description='Describe a specified dataset.')
parser.add_argument('-lang', '--language', type=str, default=None,
                    help='Choose the language of the dataset (if FARMACO, else automatically ENFERMEDAD). Choose from: es, it, en')
parser.add_argument('-d', '--dataset', type=str, default='train',
                    help='Choose the dataset. Choose from: train, dev, test')
    
args = parser.parse_args()
    
if args.language and args.language not in ['es', 'it', 'en']:
    raise ValueError("Language must be either es, it or en.")
if args.dataset not in ['train', 'dev', 'test']:
    raise ValueError("Language must be either train, dev or test.")

if args.language: #FARMACO
    path = f'../datasets/track2_converted/{args.dataset}/{args.language}/all_{args.dataset}.csv'
    ent_type = "FARMACO"
    lang = args.language
    dataset = args.dataset
else: #ENFERMEDAD
    path = f'../datasets/track1_converted/{args.dataset}/all_{args.dataset}.csv'
    ent_type = "ENFERMEDAD"
    lang = "es"
    dataset = args.dataset

df = prepare_dataset(path)

plot_relative_positions(df)

describe_cvs(path)