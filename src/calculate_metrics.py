import argparse
import pandas as pd

def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

def calculate_metrics(gold_df, pred_df):
    gold_set = set(zip(gold_df['filename'], gold_df['start_span'], gold_df['end_span'], gold_df['text']))
    pred_set = set(zip(pred_df['filename'], pred_df['start_span'], pred_df['end_span'], pred_df['text']))
    
    tp_set = gold_set & pred_set  # intersection
    fp_set = pred_set - gold_set  # predictions not in gold standard
    fn_set = gold_set - pred_set  # gold standard entries not in predictions
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, tp_set, fp_set, fn_set

def group_results_by_filename(tp_set, fp_set, fn_set):
    results_dict = {}
    
    for item in tp_set:
        filename = item[0]
        if filename not in results_dict:
            results_dict[filename] = {"tp": [], "fp": [], "fn": []}
        results_dict[filename]["tp"].append(item)
    
    for item in fp_set:
        filename = item[0]
        if filename not in results_dict:
            results_dict[filename] = {"tp": [], "fp": [], "fn": []}
        results_dict[filename]["fp"].append(item)
    
    for item in fn_set:
        filename = item[0]
        if filename not in results_dict:
            results_dict[filename] = {"tp": [], "fp": [], "fn": []}
        results_dict[filename]["fn"].append(item)
    
    return results_dict

def save_results(results_dict, output_file):
    with open(output_file, 'w') as f:
        for filename in results_dict:
            f.write(f"Filename: {filename}\n")
            f.write("True:\n")
            for item in results_dict[filename]["tp"]:
                f.write(f"{item}\n")
            
            f.write("False:\n")
            for item in results_dict[filename]["fp"]:
                f.write(f"{item}\n")
            
            f.write("Missed:\n")
            for item in results_dict[filename]["fn"]:
                f.write(f"{item}\n")
            
            f.write("\n")

def main(gold_file, pred_file, output_file, dataset):
    gold_df = load_tsv(gold_file)
    pred_df = load_tsv(pred_file)
    
    gold_df = gold_df[['filename', 'start_span', 'end_span', 'text']]
    pred_df = pred_df[['filename', 'start_span', 'end_span', 'text']]
    
    if dataset == 'test':
        mapping_df = pd.read_csv('../datasets/multicardioner_test+background_fname-mapping.tsv', sep='\t', header=None)
        gold_df['filename'] = gold_df['filename'].map(lambda x: next((j[:len(j)-4] for i, j in mapping_df.values if i[:len(i)-4] == x), x))
        print(gold_df.head())
        
    precision, recall, f1, tp_set, fp_set, fn_set = calculate_metrics(gold_df, pred_df)
    
    results_dict = group_results_by_filename(tp_set, fp_set, fn_set)
    
    save_results(results_dict, output_file)
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate precision, recall, and F1 score between gold standard and prediction TSV files.')
    parser.add_argument('pred_file', type=str, help='Path to the prediction TSV file')
    parser.add_argument('-lang', '--language', type=str, default="es",
                        help='Choose the language you want to evaluate the model on. Choose from: es, it, en')
    parser.add_argument('-t', '--type', type=str, default="ENFERMEDAD",
                        help='Choose the entity type. Choose from: ENFERMEDAD, FARMACO.')
    parser.add_argument('-o', '--output_file', type=str, default='results.txt',
                        help='Path to the output file to save results.')
    parser.add_argument('-d', '--dataset', type=str, default='test',
                        help='Calculate metrics based on dev or test set. Choose from: dev, test')
    
    args = parser.parse_args()
    
    if args.language not in ['es', 'it', 'en']:
        raise ValueError("Language must be either es, it or en.")
    
    if args.dataset not in ['test', 'dev']:
        raise ValueError("Dataset must be either test or dev.")
    
    if args.type == 'ENFERMEDAD':
        gold_file = f"../datasets/track1/cardioccc_{args.dataset}/tsv/multicardioner_track1_cardioccc_{args.dataset}.tsv"
    else:
        gold_file = f"../datasets/track2/cardioccc_{args.dataset}/{args.language}/tsv/multicardioner_track2_cardioccc_{args.dataset}_{args.language}.tsv"
    
    print(f"Checking alignment between golden standard {gold_file} and predictions {args.pred_file}...")
    
    main(gold_file, args.pred_file, args.output_file, args.dataset)
