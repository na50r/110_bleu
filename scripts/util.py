import os
from os.path import join, exists, isfile
from sentence_splitter import SentenceSplitter
from dotenv import load_dotenv
import subprocess

# Use override=True: https://docs.smith.langchain.com/observability/how_to_guides/toubleshooting_variable_caching
# Important when dealing with older API keys in Jupyter Notebook cache
load_dotenv(override=True)

# Shortened version of the same splitter used in bertalign
LANG_ISO = {
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'fi': 'Finnish',
    'fr': 'French',
    'it': 'Italian',
    'nl': 'Dutch',
    'pt': 'Portuguese',
    'sv': 'Swedish',
}


def delete_files_in_folder(folder_path: str):
    # Based on https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/
    for filename in os.listdir(folder_path):
        file_path = join(folder_path, filename)
        if isfile(file_path):
            os.remove(file_path)


def get_env_variables(*args: str) -> str | None | tuple[str | None, ...]:
    if len(args) == 1:
        return os.getenv(args[0])
    return tuple(os.getenv(arg) for arg in args)


def split_sents(text: str, lang: str) -> list[str]:
    # One-to-one implementation of split_sents in bertalign
    if lang in LANG_ISO:
        splitter = SentenceSplitter(language=lang)
        sents = splitter.split(text=text)
        sents = [sent.strip() for sent in sents]
        return sents
    else:
        raise Exception(f'The language {LANG_ISO[lang]} is not supported yet')


def store_sents(sents: list[str], folder_path: str, src_lang: str, tgt_lang: str):
    filename = f'{src_lang}-{tgt_lang}.txt'
    if not exists(folder_path):
        os.makedirs(folder_path)
    with open(join(folder_path, filename), 'w') as f:
        for sent in sents:
            print(sent.strip(), file=f)


def load_sents(folder_path: str, src_lang: str, tgt_lang: str) -> list[str]:
    filename = join(folder_path, f'{src_lang}-{tgt_lang}.txt')
    with open(filename, 'r') as f:
        data = [s.strip() for s in f.readlines()]
    return data


def get_git_revision_short_hash() -> str:
    # Code from: https://stackoverflow.com/a/21901260
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()