# Named Entity Recognition for W-2 Forms

This project aims to create a named entity recognition (NER) model for W-2 forms, which are tax documents that report the wages and taxes of employees in the United States. The model uses the distilbert-base-uncased model from the transformers library as a base, and fine-tunes it on a custom dataset of W-2 forms with annotated ner tags. The ner tags are based on the fields of the W-2 forms, such as employer name, employee name, social security number, wages, taxes, etc. The model can then be used to extract these information from new W-2 forms.

## Data

The data consists of train and validation sets of W-2 forms in tsv format. Each tsv file contains the coordinates, transcripts, and ner tags of each box in a W-2 form. The train set has 1000 tsv files, and the validation set has 200 tsv files. The validation set is divided into two parts: one with labels (val_w_ann) and one without labels (val). The data is loaded from a given path using a custom function that splits long transcripts into two parts and appends them as separate rows in a pandas dataframe.

## Tokenization

The data is tokenized using the AutoTokenizer class from the transformers library, which loads the tokenizer for the distilbert-base-uncased model. The tokenizer converts the transcripts into input ids and word ids, which are used to create ner tags for each token. The ner tags are numeric codes that correspond to the fields of the W-2 forms. The tokens and ner tags are stored as columns in pandas dataframes, and then converted to Dataset objects from the datasets library.

## Model

The model is an instance of the AutoModelForTokenClassification class from the transformers library, which loads a model for token classification from a pretrained model name. The model is initialized with 16 labels, and id2label and label2id dictionaries that map the numeric codes to the ner tags and vice versa. The model is moved to the device (cuda or cpu) depending on its availability.

## Training

The model is trained using the Trainer class from the transformers library, which takes the model, the training arguments, the train and validation datasets, the tokenizer, the data collator, and the metric as inputs. The training arguments specify various hyperparameters and settings for training and evaluation, such as learning rate, batch size, number of epochs, weight decay, evaluation strategy, save strategy, etc. The data collator is an instance of the DataCollatorForTokenClassification class from the transformers library, which handles padding and masking of the inputs. The metric is an instance of the seqeval metric from the datasets library, which evaluates the model on various aspects such as precision, recall, f1-score, and accuracy.

## Inference

The model can be used to make predictions on new W-2 forms using the pipeline function from the transformers library. The pipeline function creates a classifier object using the ner pipeline, the trained model, the tokenizer, and the device. The classifier object can take a list of words as input and return a list of dictionaries with tokens and predicted entities. A custom function is defined to extract labels from inference output and append them to a list. Another custom function is defined to generate validation tsv files with predicted labels for each transcript in a given directory.

## Conclusion

This project demonstrates how to create a named entity recognition model for W-2 forms using PyTorch and transformers. The model can be used to extract useful information from tax documents and automate tasks such as data entry or analysis.
