# MultiCardioNER Dataset

MultiCardioNER is a shared task about the adaptation of clinical NER systems to the cardiology domain. It uses a combination of two existing datasets ([DisTEMIST](https://temu.bsc.es/distemist) for diseases and the newly-released DrugTEMIST for medications), as well as a new, smaller dataset of cardiology clinical cases annotated using the same guidelines.

Participants are provided DisTEMIST and DrugTEMIST as training data to use as they see fit (1,000 documents, with the original partitions splitting them into 750 for training and 250 for testing). The cardiology clinical cases (cardioccc) are meant to be used as a development or validation set (258 documents). Another set of cardioccc will be released later on for testing.

MultiCardioNER proposes two tasks:

- Track 1: Spanish adaptation of disease recognition systems to the cardiology domain.
- Track 2: Multilingual (Spanish, English and Italian) adaptation of medication recognition systems to the cardiology domain.

MultiCardioNER was developed by the Barcelona Supercomputing Center's NLP for Biomedical Information Analysis and used as part of BioASQ 2024. For more information on the corpus, annotation scheme and task in general, please visit: https://temu.bsc.es/multicardioner.

## Folder Structure

This repository includes:

- `track1/`: Data for the task's Track 1 (Spanish adaptation of disease recognition systems). It includes two subfolders: `distemist_train` (complete DisTEMIST dataset to be used as training set) and `cardioccc_dev` (cardiology clinical cases to be used as development set).
- `track2/`: Data for the task's Track 2 (Multilingual adaptation of medication recognition systems). It includes two subfolders: `drugtemist_train` (complete DrugTEMIST dataset to be used as training set) and `cardioccc_dev` (cardiology clinical cases to be used as development set). In turn, each of these folders contain the separated data in each language (`en` for English, `es` for Spanish and `it` for Italian).

## Data Format

The MultiCardioNER corpus is offered in two different formats, each separated in a different folder:

- `brat/`

Original documents resulting from the annotation process using the brat tool. Includes the brat .ann files together with the .txt files. For more information on brat's format please visit: https://brat.nlplab.org/standoff.html.

- `tsv/`

This folder includes tab-separated files (tsv) where each line represents an annotation. Each of them has the following columns: "filename", "ann_id" (annotation identifier, not really needed for the task but kept for traceability), "label", "start_span", "end_span" and "text".

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Contact

If you have any questions or suggestions, please contact us at:

- Salvador Lima-López (<salvador [dot] limalopez [at] gmail [dot] com>)
- Martin Krallinger (<krallinger [dot] martin [at] gmail [dot] com>)
