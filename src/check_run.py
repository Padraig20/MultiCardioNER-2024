import argparse
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def extract_text_from_filename(filename, foldername):
    #with open(os.path.join(f"../datasets/test+background/{lang}", filename + ".txt"), 'r', encoding='utf-8') as file:
    #    content = file.read().replace('\n', ' ')
    with open(os.path.join(foldername, filename + ".txt"), 'r', encoding='utf-8') as file:
        content = file.read().replace('\n', ' ')
    return content

def check_alignment(row, text):
    start_span = row['start_span']
    end_span = row['end_span']
    extracted_text = row['text']
    aligned_text = text[start_span:end_span]
    return aligned_text == extracted_text

def get_aligned_text(row, text):
    start_span = row['start_span']
    end_span = row['end_span']
    aligned_text = text[start_span:end_span]
    return aligned_text

def get_extracted_text(row):
    extracted_text = row['text']
    return extracted_text

def align_text(row, text):
    if not check_alignment(row, text):
        start_span = row['start_span'] - 1
        end_span = row['end_span']
        if text[start_span:end_span] == row['text']:
            return start_span, end_span
        start_span = row['start_span']
        end_span = row['end_span'] + 1
        if text[start_span:end_span] == row['text']:
            return start_span, end_span
        start_span = row['start_span'] + 1
        end_span = row['end_span']
        if text[start_span:end_span] == row['text']:
            return start_span, end_span
    print("Alignment check failed for the following row:")
    print(f"{row['actual_text']} -- ACTUAL")
    print(f"{row['extracted_text']} -- EXTRACTED\n")
    return row['start_span'], row['end_span']

def main(input_file, foldername):
    data = pd.read_csv(input_file, sep='\t', encoding='utf-8')
    
    filename_texts = {}
    
    for filename in data['filename'].unique():
        file_text = extract_text_from_filename(filename, foldername)
        filename_texts[filename] = file_text
    
    data['alignment'] = data.apply(lambda row: check_alignment(row, filename_texts[row['filename']]), axis=1)
    data['actual_text'] = data.apply(lambda row: get_aligned_text(row, filename_texts[row['filename']]), axis=1)
    data['extracted_text'] = data.apply(lambda row: get_extracted_text(row), axis=1)
    
    failed_spans = data[data['alignment'] == False]
    if not failed_spans.empty:
        print("Alignment check failed for the following spans:")
        print(failed_spans[['filename', 'label', 'start_span', 'end_span', 'text', 'actual_text', 'extracted_text']])
        
        print("Attempting to correct alignment...\n")

        failed_spans['start_span'], failed_spans['end_span'] = zip(*failed_spans.apply(lambda row: align_text(row, filename_texts[row['filename']]), axis=1))

        corrected_spans = data[data['alignment'] == True]
        combined_spans = pd.concat([corrected_spans, failed_spans])
    
        output_file = f"{input_file[:-4]}_corrected.tsv"
        selected_columns = ['filename', 'label', 'start_span', 'end_span', 'text']
        combined_spans[selected_columns].to_csv(output_file, sep='\t', index=False, header=True)
        
        print(f"Data written to {output_file}")
        
        input_rows = len(data)
        output_data = pd.read_csv(output_file, sep='\t')
        output_rows = len(output_data)

        if input_rows == output_rows:
            print("Data correction was successful, input and output files have the same amount of rows.")
        else:
            print(f"Data correction failed for some rows! Input file and output file have different amount of rows: {input_rows} vs {output_rows}")
    else:
        print("Alignment check passed for all spans.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check alignment of spans in TSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file')
    parser.add_argument('-lang', '--language', type=str, default="es",
                    help='Choose the language you want to evaluate the model on. Choose from: es, it, en')
    parser.add_argument('-t', '--type', type=str, default="ENFERMEDAD",
                        help='Choose the entity type. Choose from: ENFERMEDAD, FARMACO.')
    parser.add_argument('-d', '--dev', type=bool, default=False,
                        help='Whether to use the development set or not. Default is False.')
    
    args = parser.parse_args()
    
    if args.language not in ['es', 'it', 'en']:
        raise ValueError("Language must be either es, it or en.")
    
    if not args.dev:
        folder_name = f"../datasets/test+background/{args.language}"
    else:
        if args.type == 'ENFERMEDAD':
            folder_name = "../datasets/track1/cardioccc_dev/brat/"
        else:
            folder_name = f"../datasets/track2/cardioccc_dev/{args.language}/brat/"
    
    args = parser.parse_args()

    if args.language not in ['es', 'it', 'en']:
        raise ValueError("Language must be either es, it or en.")

    args = parser.parse_args()
    
    main(args.input_file, folder_name)
