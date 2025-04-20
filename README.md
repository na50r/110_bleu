# Project
Inspired by Philip Kohen's BLEU score matrix in [this paper](https://aclanthology.org/2005.mtsummit-papers.11/), we try to recreate a similar matrix by evaluating modern MT systems, namely DeepL and GPT4.1. This requires us to set some rules for ourselves, choose the respective datasets and define configurations for the respective systems and set them in stone before the translation process; so no change of translation or pre-processing or data management related code mid-translation. Mainly because translation costs money. 

# Installation
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

## Environment Variables
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
* Variables suffixed with `_STORE` are required to store dataset files locally on your machine


# Data Managers
* Contains dataset wrappers, makes it easier to get sentence pairs from the respective datasets
* `EPManager` is a wrapper around [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)
* `FloresPlusManager` is a wrapper around [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
* `Opus100Manager` is a wrapper around [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

## Usage
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

# Translators
* Functions that are used for translation tasks

# Utility
* Contains utility functions used for various purposes
## Logging
* `MyLogger` and `Log` are used to log each translation from source to target.
* Logging is done to ensure documentation of each translation, especially in cases where translations have to be re-run
* TikToken with `gpt-4o` encoder is used for both GPT and DeepL translation in terms of logging, only the actual input and output texts are tokenized and counted, so we can prove that the same text was used as input for GPT and Deepl
* GPT will always have slightly more input due to the prompt which shall not vary besides the language pairs

