import pandas as pd
from googletrans import Translator
import uuid
import time
import os

file_path = 'mtsamples_en.csv'
data = pd.read_csv(file_path)
cv_data = data[(data['medical_specialty'] == ' Cardiovascular / Pulmonary') & data['keywords'].notna()]

translator = Translator()

with open ('tsv/mtsamples_es.tsv', 'a') as tsv_file:
    tsv_file.write("filename\tann_id\tlabel\tstart_span\tend_span\ttext\n")

counter = 1

for index, row in cv_data.iterrows():
    print(f"Processing row {counter} from {cv_data.shape[0]}...")
    counter = counter + 1
    
    keywords = row['keywords'].split(', ')
    keywords = [keyword for keyword in row['keywords'].split(', ') if keyword or keyword == 'cardiovascular / pulmonary']
    transcription = row['transcription']
    
    try:
        translated_transcription = translator.translate(transcription, dest='es').text
    except Exception as e:
        print(f"Error: {e}")
        print(f"Transcription: {transcription}")
        print("Skipping...")
        continue
    translated_keywords = []
    for keyword in keywords:
        try:
            translated_keyword = translator.translate(keyword, dest='es').text
            translated_keywords.append(translated_keyword)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Keyword: {keyword}")
    
    # Generate BRAT files

    file_uuid = str(uuid.uuid4())

    txt_file_path = f'brat/{file_uuid}.txt'
    ann_file_path = f'brat/{file_uuid}.ann'

    with open(txt_file_path, 'w') as txt_file, open(ann_file_path, 'w') as ann_file:
        txt_file.write(translated_transcription)
            
        # Generate annotations for each keyword
        for keyword in translated_keywords:
            start_index = translated_transcription.find(keyword)
            end_index = start_index + len(keyword)
            
            if start_index == -1:
                continue
            
            # Write annotation to ann file
            ann_file.write(f'T\tENFERMEDAD\t{start_index}\t{end_index}\t{keyword}\n')
            
            with open ('tsv/mtsamples_es.tsv', 'a') as tsv_file:
                tsv_file.write(f'{file_uuid}\tT\tENFERMEDAD\t{start_index}\t{end_index}\t{keyword}\n')
    
    time.sleep(0.5)

print("Done! Now cleaning up...")

ann_files = [file for file in os.listdir('brat') if file.endswith('.ann')]

for ann_file in ann_files:
    txt_file = ann_file.replace('.ann', '.txt')
    # delete empty files
    if os.path.getsize(f'brat/{ann_file}') == 0:
        os.remove(f'brat/{ann_file}')
        os.remove(f'brat/{txt_file}')