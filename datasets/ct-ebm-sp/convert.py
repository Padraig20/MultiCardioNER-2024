import json

def convert_line_to_json(line):
    json_data = json.loads(line)
    return json_data

def read_file_and_convert_to_json(input_path, output_path):
    with open(input_path, 'r') as file:
        for line in file:
            json_data = convert_line_to_json(line)
            tokens = json_data['tokens']
            labels = json_data['ner_tags']
            with open(output_path, 'a') as file:
                for token, label in zip(tokens, labels):
                    file.write(f"{token}\t{label}\n")
                file.write('\n')

# Example usage
input_path = './test/test.json'
output_path = './test/test.conll'
read_file_and_convert_to_json(input_path, output_path)