import re
import pandas as pd

# Load your data
data = pd.read_csv('all_train.csv', delimiter='|')

# Regex pattern for tokenization
token_pattern = r"\w+|\w+(?='s)|'s|['\".,!?;]"

# Path for the output file
output_path = 'all_train.conll'

# Process and write to file
with open(output_path, 'w') as file:
    for index, row in data.iterrows():
        # Tokenizing the sentence based on the provided regex pattern
        tokens = re.findall(token_pattern, str(row.iloc[0]))  # Adjust the column index as necessary

        # Splitting tags normally as they are aligned with tokens
        tags = str(row.iloc[1]).split()  # Adjust the column index as necessary

        # Write each token and its corresponding tag to the file
        for token, tag in zip(tokens, tags):
            file.write(f"{token}\t{tag}\n")
        
        # Add a newline after each sentence to denote a new data entry
        file.write("\n")

print("Conversion to CoNLL format completed.")

