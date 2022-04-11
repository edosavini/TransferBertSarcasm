# TransferBertSarcasm
Code of the paper Intermediate-Task Transfer Learning with BERT for Sarcasm Detection

# Installation
The code has been tested for Python 3.6. Install the required libraries by running the following command in your local environment:

```commandline
pip install -r constraints.txt
```

# Experiments with no intermediate pretraining
Experiments with simple BERT and no intermediate pretraining can be launched with:

```commandline
python3 scripts/train_local.py train <path-to-config-file.json> -s <output-folder-with-weights-and-metrics> --include-package <your-own-library> 
```
For example, this command will train and test a BERT sarcasm classifier on some sample data that we uploaded from SARCTwitter:
```commandline
python3 scripts/train_local.py train config/SARCTwitter/config.json -s BertExample --include-package my_library
```
The classifiers and the data readers are implemented in the directory `my_library` and selected by the `config.json` file, 
where the whole configuration of the experiment is defined, along with the input parameters, 
the optimizer, the model structure, the input data path, etc.

# Experiments with intermediate pretraining
In order to perform transfer learning experiments, use the `transfer` keyword and add the path to the weights 
of the pretrained intermediate model.
```commandline
python3 scripts/train_local.py transfer config/SARCTwitter/config.json -s SarcTwitt_IMDB --transfer-model model/IntermediateIMDB.th --include-package my_library
```

# Need more?
Thank you for your interest in this research. To request all the data used in our experiments or the models' weights, 
please write an email to edoardosavini95@gmail.com

# Copyright

When using this data in your research, please reference the following publication:

Savini, E.; Caragea, C. "[Intermediate-Task Transfer Learning with BERT for Sarcasm Detection.](https://doi.org/10.3390/math10050844)" Mathematics 2022, 10, 844.
