
"""
  Class to convert CoNLL NER data to JSON or JSONLine that can be used with different annotation tools like doccano or
  Label Studio
  
  NER data is expected to be in the following format. New line is considered as sentence breaker. Paragraphs and documents are
  not taken care of but can be treated quite easily
  '''
    010 I-MISC
    is O
    the O
    tenth O
    album O
    from O
    Japanese I-MISC
    Punk O
    Techno O
    band O
    The I-ORG
    Mad I-ORG
    Capsule I-ORG
    Markets I-ORG
    . O
    
  '''
  
  Formated data saved in JSON file will look like this
  {"text": "010 is the tenth album from Japanese Punk Techno band The Mad Capsule Markets .", "labels": [[0, 4, "I-MISC"], [28, 37, "I-MISC"], [54, 78, "I-ORG"]]}
  
  ## USAGE
  converter = CONLL2JSON()
  converter.parse(input_file, output_file, sep, fmt)
  
"""

import pandas as pd
import json


class CONLL2JSON:
    def __init__(self):
        self._sep_list = [' ', '\t']
        self._fmt_list = ['json', 'jsonl']

    @staticmethod
    def _load_text(txt_path):
        """
        Opens the container and reads file
        Returns a list[string]
        :param txt_path: filepath
        """
        with open(txt_path, 'rt') as infile:
            content = infile.readlines()

        return content

    @staticmethod
    def _process_text(content, sep):
        """
        given a list of txt_paths
        -process each
        :param content: list of strings
        :param sep: string representing separator
        :return: list of dicts
        """
        list_dicts = []
        words = []
        # detect newline and make sentences
        for line_num, line in enumerate(content):
            if line != '\n':
                words.append(line.strip('\n'))
            else:
                sentence = " ".join([word.split(sep)[0] for word in words])
                words = [word.split(sep) for word in words]

                df_words = pd.DataFrame(data=words, columns=['word', 'ner'])

                m = df_words['ner'].eq('O')

                try:

                    df_words_filtered = (df_words[~m].assign(Label=lambda x: x['ner'].str.replace('^[IB]-', ''))
                                         .groupby([m.cumsum(), 'ner'])['word']
                                         .agg(' '.join)
                                         .droplevel(0)
                                         .reset_index()
                                         .reindex(df_words.columns, axis=1))
                except ValueError:
                    words = []
                    continue
                label_w_pos = []
                for row in json.loads(df_words_filtered.to_json(orient='records')):
                    start_pos = sentence.find(row['word'])
                    end_pos = start_pos + len(row['word']) + 1

                    label_w_pos.append([start_pos, end_pos, row['ner']])

                list_dicts.append({'text': sentence, 'labels': label_w_pos})

                words = []
        return list_dicts

    @staticmethod
    def _write_text(list_dicts, fmt, output_file):
        """
        :param list_dicts: list of dicts formatted in json
        :param fmt: format of json file either JSON object or JSON Line file
        :param output_file: file to save data
        :return:
        """

    def parse(self, input_file: str, output_file: str, sep: str, fmt) -> None:
        if sep not in self._sep_list:
            raise RuntimeError(f'Separator should be in "{self._sep_list}", provided separator was "{sep}"')

        if fmt not in self._fmt_list:
            raise RuntimeError(f'Format should be in "{self._fmt_list}", provided file format was "{fmt}"')

        content = self._load_text(input_file)
        list_dicts = self._process_text(content, sep)

        if fmt == 'json':
            data = json.dumps(list_dicts)
        else:
            list_dicts = [json.dumps(l) for l in list_dicts]
            data = '\n'.join(list_dicts)

        with open(output_file, 'w') as json_file:
            json_file.write(data)

converter = CONLL2JSON()
converter.parse('dev/all_dev.conll', 'tmp.json', '\t', 'json')
