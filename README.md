# Project

This repository is accompanying a Bachelor thesis on machine translation evaluation, it contains the code used to obtain translations and evaluate them.

Inspired by the BLEU score matrix shown in Phillip Koehn's [Europarl Statistical Machine Translation paper](https://aclanthology.org/2005.mtsummit-papers.11/), we try to recreate a similar matrix by obtaining translations from two modern machine translation systems, namely DeepL and GPT4.1 and evaluating them. 

**NOTE**:
This codebase is NOT an API nor a Python Package. Type annotation and docstrings were added merely for readability and intepretability's sake, for project evaluators. Only relevant functions/methods and classes received them, whereas private methods or utility function should be understandable by just reading the code (variable names, see how their used, etc.)

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

Then make sure that this folder exists

```sh
mkdir C:\Files\Storage\Flores
```

## Data Managers

* Contains dataset wrappers, makes it easier to get sentence pairs from the respective datasets
* Every DataManager must implement a `get_sentence_pairs` method, how it is implemented can differ based on how the HuggingFace dataset is structured/should be used.
    - `EuroParlManager` is a wrapper around [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl)
    - `FloresPlusManager` is a wrapper around [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
    - `Opus100Manager` is a wrapper around [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

* **Note:** All datasets support more language pairs than required for this project but here, we restrict ourselves to 110 pairs of the 11 European languages used in Phillip Koehn's paper, namely: English, German, Danish, Greek, Spanish, Portuguese, Dutch, Swedish, French, Italian, Finnish.
    - The `Opus100Manager` is English-centric, thus only supports a subset of 20 pairs

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
* We enforce all translator clients to have the same method,  `translate_document`
* The implementation of it differs based on what the API / Python packages provide but the input and output of `translate_document` should be the same.
* In other parts of this code and for the main translation task, we just call `translate_document`

### MockClient

* We also implemented a `MockClient` for testing purposes. This client provides additional features such as specifying for which language pair or translation count to fail/raise an error.
* The use of it can be found in `test_tasks.py`

## Task

* `task.py` implements translators and data managers to formulate translation tasks
* A translation task is defined as the translation of a set of pairs *P* from a dataset *d* using a translator *t*, more formally we:

* Define a set of languages as `L = {de, fr, da, el, es, pt, nl, sv, en, it, fi}`
* `P ⊆ {(x, y) | x ∈ L, y ∈ L, x ≠ y}`
* `d ∈ {europarl, flores_plus, opus-100}`
* `t ∈ {deepl_document, gpt-4.1-2025-04-14}`
* Then a task `(P, d, t)` means: *Translate all pairs in **P** using their resp. sentences from **d** with **t***

* In addition to this, we can define 
    - how many sentences we want to translate
    - how often we allow to call the API again 
    - what conditions we consider to accept or reject translations in terms of number of translated sentences
    - if the task is a manual retry task or not (in case a prior task failed to deliever desirable translations despite automatic retry and we have to run it again manually)

## Procedure

* `procedure.py` just allows transparent use of `task.py` by creating the tasks for the selected procedure and allowing the user to run the task with CLI or Jupyter Notebook. It also accounts for loggings.
* Note: `procedure.py` is never used directly, we inherit the Procedure class and define procedures outside the scripts folder
* An example of a procedure could be:
    - `P = {(x, y) | x ∈ L, y ∈ L, x = en ∨ y = en}`
    - `D = {europarl, opus-100}`
    - `T = {gpt-4.1-2025-04-14}`

* *TL; DR*: Translate all pairs that include English as src or tgt using GPT4.1 using sentences from the EuroParl and Opus100 datasets.
    - The procedure defines two tasks: `(P, europarl, gpt-4.1-2025-04-14)` and `(P, opus-100, gpt-4.1-2025-04-14)`   
    - Thus there will be only two cli commands the user can run to run these tasks
        

        ```sh
        python proc.py task --model gpt-4.1-2025-04-14 --dataset opus-100
        python proc.py task -m gpt-4.1-2025-04-14 -d europarl
        ```
        
* In addition, the user can view their possible inputs:
```sh
python proc.py inputs
```

```
Available datasets: ['europarl', 'opus-100']
Available models: ['gpt-4.1-2025-04-14']
```

* Or view the details of a task:

```sh
python proc.py task -m gpt-4.1-2025-04-14 -d europarl
```

```
Task details for europarl - gpt-4.1-2025-04-14:
  id: 2a9a89c4-813e-4285-a8ef-55d0815a18b9
  store: tmp\proc\europarl\gpt-4.1-2025-04-14
  pairs: [('en', 'fi'), ('en', 'fr'), ('de', 'en'), ('es', 'en'), ('en', 'el'), ('fi', 'en'), ('en', 'pt'), ('en', 'da'), ('en', 'nl'), ('fr', 'en'), ('en', 'es'), ('da', 'en'), ('it', 'en'), ('en', 'sv'), ('nl', 'en'), ('pt', 'en'), ('en', 'it'), ('sv', 'en'), ('el', 'en'), ('en', 'de')]
  dm: <scripts.data_management.EuroParlManager object at 0x000001CDC00AC7D0>
  tl_logger: <scripts.logger.TranslationLogger object at 0x000001CDC01DF4A0>
  num_of_sents: 10
  client: <scripts.translators.GPTClient object at 0x000001CDC020C560>
  acceptable_range: (5, 15)
  manual_retry: False
  max_retries: 3
  retry_delay: 0
```

* For more use cases, please view [Demo.ipynb](https://github.com/na50r/110_bleu/blob/main/Demo.ipynb)

## Util

* Contains utility functions used for various purposes

## Testing

Simple unit tests were implemented to confirm the functionality of the data managers and the translation task as a whole. 

```sh
python -m pytest test_tasks.py test_datamanagers.py
```

* If you want to see the logs, run:

```sh
python -m pytest -o log_cli=true -o log_cli_level=INFO test_tasks.py
```

* If you want to run individual tests, run:

```sh
python -m pytest test_tasks.py::test_logging_with_scenario_B
```

## Logging

* This code employs Python's `logging` package to log information of translation tasks.
* Logs are shown in terminal / Jupyter Notebook cell where the task is run and but also stored in a `.log` file.
* The `.log` file contains debug level logs, more details, grep can be used to filter them on the prefixes as desired
    - Use Regex in double quotes if you work on Windows 

### TranslationLogs

* In addition to regular logs, we also store translation specific information in a JSONL file.
* This information is per translation as opposed to per task and contains additionally:
    - number of input and output lines (Translation input/output is text separated by newline characters)
    - number of input and output sentences (Using language specific [SentenceSplitter](https://github.com/mediacloud/sentence-splitter))
    - number of input and output characters (Using `len`)
    - number of input and output tokens (Using [tiktoken](https://github.com/openai/tiktoken) with a tokenizer/encoder based on gpt-4o)
    - number of input and output model tokens (Only for GPT4.1, found in the response body of the API)
    - start and end time of translation (Unix time)
* This information is stored for analysis purposes, such as computing the exact time per translation or comparing estimated tokens to actual tokens (again, only possible for GPT4.1 as DeepL computes per characters)
