# MultiCardioNER - 2024

MultiCardioNER is a shared task about the adaptation of clinical NER systems to the cardiology domain. It uses a combination of two existing datasets ([DisTEMIST](https://temu.bsc.es/distemist) for diseases and the newly-released DrugTEMIST for medications), as well as a new, smaller dataset of cardiology clinical cases annotated using the same guidelines.

Participants are provided DisTEMIST and DrugTEMIST as training data to use as they see fit (1,000 documents, with the original partitions splitting them into 750 for training and 250 for testing). The cardiology clinical cases (cardioccc) are meant to be used as a development or validation set (258 documents). Another set of cardioccc will be released later on for testing.

MultiCardioNER proposes two tasks:

- Track 1: Spanish adaptation of disease recognition systems to the cardiology domain.
- Track 2: Multilingual (Spanish, English and Italian) adaptation of medication recognition systems to the cardiology domain.

MultiCardioNER was developed by the Barcelona Supercomputing Center's NLP for Biomedical Information Analysis and used as part of BioASQ 2024. For more information on the corpus, annotation scheme and task in general, please visit: https://temu.bsc.es/multicardioner.

## Track 1 - Disease Recognition in Spanish

Since this model only contains Spanish words, we will use language specific BERT models.

## Data Loading

The dataloader.py file contains two classes: DataLoader and Custom_Dataset.

The DataLoader class is responsible for loading and preprocessing the dataset. It reads the dataset from a TSV file, drops unnecessary columns, and gets a list of unique filenames. It then initializes a tokenizer from the HuggingFace transformers library and adds some custom tokens. The data can be split into training, validation, and test sets, or returned as a whole depending on the full parameter of the load_dataset method.

The Custom_Dataset class is a subclass of PyTorch's Dataset class. It is used for loading and tokenizing sentences on-the-fly. This class is designed to be used with a PyTorch DataLoader to efficiently load and preprocess data in parallel.

Together, these classes provide a convenient way to handle data loading and preprocessing for a machine learning model.

## Ideas for Pre-Training

First of all, we need to adapt the model to the very specific corpus of medical texts in Spanish. In order to expand the vocabulary of the model via domain-specific pretraining, it would be wise to use [MedLexSp](https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-022-00281-5), an only recently released dataset containing curated medical vocabulary in Spanish. Furthermore, to increade the model's understanding of patient notes, we could use masked language modelling in patient admission notes of the TREC CT proceedings. These texts would need to be automatically translated into Spanish, possible via DeepL.

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.