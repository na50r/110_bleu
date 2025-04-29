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
* `EPManager` is a wrapper around [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)
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
langs = ['de', 'da', 'el', 'es',
'pt', 'nl', 'fi', 'sv', 'fr', 'it', 'en']
pairs = []
for s in langs:
    for t in langs:
        if s!=t:
            pairs.append((s, t))

# Or use
pairs = [tuple(pair.split('-')) for pair in EuroParlManager.EP_PAIRS]
mirrored = [(pair[1], pair[0]) for pair in pairs]
pairs = pairs + mirrored

dm = EuroParlManager()
# Will download data required for all 110 pairs from all 3 datasets
for pair in pairs:
    dm.get_sentence_pairs(pair[0], pair[1], num_of_sents=10)
```

## Translators
* Functions that are used for translation tasks
* Usage of translators can be seen in `Demo.ipynb` in combination with the rest of the code. 

## Util
* Contains utility functions used for various purposes

## Logging

* `MyLogger` and `Log` are used to log each translation from source to target.
* Logging is done to ensure documentation of each translation, especially in cases where translations have to be re-run
* TikToken with `gpt-4o` encoder is used for both GPT and DeepL translation in terms of logging, only the actual input and output texts are tokenized and counted, so we can prove that the same text was used as input for both translators
* GPT will always have slightly more input due to the prompt which shall not vary besides the language pairs
* Logger also accounts for re-running, in case translation calls have to be run again, it allows us to link the new log to the old one using logged ids

## Testing
Simple unit tests were implemented to confirm the functionality of the data managers and the translation task as a whole. 
* The translation task test will not be using any API, instead it'll run it through a MockClient that uses the Ceaser Cipher to mock translation
* The translation task test should not store anything on your machine, the logs will be printed in the terminal and generated foldes/files will be deleted after test execution.
* The data manager test will download required files if missing, please make sure to have set the respective environment variables in the `.env` file. 
```
python -m pytest -s test_datamanagers.py test_translation_task.py
```