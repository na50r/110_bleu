# Project
Inspired by Philip Kohen's BLEU score matrix in [this paper](https://aclanthology.org/2005.mtsummit-papers.11/), we try to recreate a similar matrix by evaluating modern MT systems, namely DeepL and GPT4.1. This requires us to set some rules for ourselves, choose datasets, define configurations for the respective systems and set them in stone before the translation process; so no change of translation, pre-processing or data management related code mid-translation. Mainly because translation costs money. We try to ensure this by keeping this code readable and documenting each translation task automatically, including the commit hash. 

**NOTE**:
This codebase is NOT an API nor a Python Package. Type annotation and docstrings were added merely for readability and intepretability's sake, for project evaluators. The most relevant functions and classes received them, whereas private methods or utility function should be understandable by just reading the code (variable names, see how their used, etc.)

## Installation
If you want to install it directly, you may use Conda:

```sh
conda env create -f environment.yml
conda activate thesis
pip install -r requirements.txt
```

* If you want to use Bertalign, resp. `align_src_mt_sents`, then clone [this fork](https://github.com/na50r/bertalign) and run:

```sh
pip install bertalign/
```

* Note, if you follow this installation process, you should have everything at the same state I have. However, if you only want to inspect parts of the code, you can also selectively install packages yourself. 
* `unbabel-comet` and `bert-score` are present in this code but not used on local machines, instead I would just uploud `scoring.py` to Google Colab and run it there. Hence not part of `requirements.txt`

### Environment Variables
* Create a `.env` file within the cloned repository and specify values for the following environment variables:

```sh
# API keys
DEEPL_API_KEY=
OPENAI_API_KEY=
HUGGINGFACE_KEY=

# Paths to folders
FLORES_STORE=
OPUS_100_STORE=
EUROPARL_STORE=
```
* DeepL and OpenAI API keys are required for the translators to work.
* HuggingFace key is required to access Flores+ dataset for the first time, downloading it
* Variables suffixed with `_STORE` are required to store dataset files locally on your machine, specify paths to folders, if you write:
```sh
FLORES_STORE=C:\Files\Storage\Flores
```
Then make sure that Flores folder exists
```sh
mkdir C:\Files\Storage\Flores
```

## Data Managers
* Contains dataset wrappers, makes it easier to get sentence pairs from the respective datasets
* `EuroParlManager` is a wrapper around [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)
* `FloresPlusManager` is a wrapper around [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
* `Opus100Manager` is a wrapper around [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

### Usage
* Getting sentence pairs from a dataset:

```py
from scripts.data_management import Opus100Manager
dm = Opus100Manager()
de_sents, en_sents = dm.get_sentence_pairs('de', 'en', num_of_sents=100)
de_sents[:5], en_sents[:5]
```

```
Files for de must be downloaded.
OPUS-100 data for de has been stored.
(['04:26:35',
  'Prähistorische Archäologie im Dritten Reich".',
  'Die Nutzungsbedingungen werden durch das Klicken des Nutzers auf "Profil Speichern" vereinbart.',
  'Ich wollte dir erst noch etwas zeigen.',
  'Du musstest wegen Shinkichi leiden.'],
 ['04:26:35',
  'Prähistorische Archäologie im Dritten Reich".',
  "By clicking on 'Save profile', you the user agree to these terms and conditions.",
  'I wanted to show you something first.',
  'You have suffered because of Shinkichi.'])
```

* If files have been already stored, nothing will be downloaded.

### Local Storage
* Data is downloaded if it is not stored locally. The default size is set to 500, so only the first 500 entries are stored locally. If you want to first download everything and then play with the methods, run:

```py
from scripts.data_management import EuroParlManager
pairs = EuroParlManager.get_pairs()
dm = EuroParlManager()
# Will download data required for all 110 pairs from all 3 datasets
for pair in pairs:
    dm.get_sentence_pairs(pair[0], pair[1], num_of_sents=10)
```

## Translators
* Defines an uniform interfaces for the external APIs. 
* We enforce all translator clients to have the same method, `translate_document`
* The implementation of it differs based on what the API / Python packages provide but the input and output of `translate_document` should be the same.
* In other parts of this code and for the main translation task, we just call `translate_document`

### MockClient
* We also implemented a `MockClient` for testing purposes. This client provides additional features such as specifying for which language pair or translation count to fail/raise an error.
* The use of it can be found in `test_tasks.py`

## Task
* `task.py` implements translators and data managers to formulate translation tasks
* A translation task is defined as the translation of a set of pairs $P$ from a dataset $d$ using a translator $t$, more formally we:

* Define a set of languages as `L = {de, fr, da, el, es, pt, nl, sv, en, it, fi}`
* `P ⊆ {(x, y) | x ∈ L, y ∈ L, x ≠ y}`
* `d ∈ {EuroParl, FloresPlus, Opus100}`
* `t ∈ {DeepL, GPT4.1}`
* Then a task `(P, d, t)` mean: *Translate all pairs in **P** using sentences from **d** with **t***

* In addition to this, we can define 
    * how many sentences we want to translate
    * how often we allow to call the API again 
    * what conditions we consider to accept or reject translations in terms of number of translated sentences
    * if the task is a manual retry task or not (in case a prior task failed to deliever desirable translations despite automatic retry and we have to run it again manually)

## Procedure
* `procedure.py` just allows transparent use of `task.py` by creating the tasks for the selected procedure and allowing the user to run the task with CLI interface or Jupyter Notebook. It also accounts for loggings.
* An example of a procedure could be:
    * `P = {(x, y) | x ∈ L, y ∈ L, x = en ∨ y = en}`
    * `D = {EuroParl, Opus100}`
    * `T = {GPT4.1}`

* *TL;DR*: Translate all pairs that include English as src or tgt using GPT4.1 using text from the EuroParl and Opus100 datasets.
    * The procedure defines two tasks: `(P, EuroParl, GPT4.1)` and `(P, Opus100, GPT4.1)`   
    * Thus there will be only two cli commands the user can run to run these tasks
        ```sh
        python proc.py run --model gpt-4.1 --dataset europarl
        python proc.py run --model gpt-4.1 --dataset opus100
        ```	

## Util
* Contains utility functions used for various purposes

## Testing
Simple unit tests were implemented to confirm the functionality of the data managers and the translation task as a whole. 
```
python -m pytest test_tasks.py test_translation_task.py
```
* If you want to see the logs, run:
```
pytest -o log_cli=true -o log_cli_level=INFO test_tasks.py
```
