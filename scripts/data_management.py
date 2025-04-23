import os
import json
from scripts.util import get_env_variables, delete_files_in_folder
from os.path import join
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod


class DataManager(ABC):
    def __init__(self):
        self.name = None
        self.split = None
    
    @abstractmethod
    def get_sentence_pairs(self, src_lang: str, tgt_lang: str, num_of_sents: int) -> tuple[list[str], list[str]]:
        pass


class FloresPlusManager(DataManager):
    # NOTE: Every sentence is aligned with every other sentence, less data overall
    EURO_LANGS = {
        'de': 'deu_Latn',
        'fr': 'fra_Latn',
        'da': 'dan_Latn',
        'el': 'ell_Grek',
        'es': 'spa_Latn',
        'pt': 'por_Latn',
        'nl': 'nld_Latn',
        'sv': 'swe_Latn',
        'en': 'eng_Latn',
        'it': 'ita_Latn',
        'fi': 'fin_Latn',
    }

    def __init__(self, split: str = "dev", size: int = 500):
        self.store = get_env_variables('FLORES_STORE')
        self.split = f'{split}[:{size}]'
        self.langs = FloresPlusManager.EURO_LANGS
        self.name = "openlanguagedata/flores_plus"

    def _same_split(self):
        split_file = join(self.store, 'split')
        try:
            with open(split_file, 'r') as f:
                split = f.readline()
        except FileNotFoundError:
            return False
        return split.strip() == self.split

    @staticmethod
    def _hugging_face_login():
        from huggingface_hub import login
        hug_key = get_env_variables('HUGGING_FACE_KEY')
        login(token=hug_key, add_to_git_credential=True)

    def _download_data(self, lang: str) -> Dataset:
        missing = self.langs[lang]
        print(f'Files for {lang} must be downloaded.')

        self._hugging_face_login()
        dataset = load_dataset(self.name, missing, split=self.split)
        return dataset

    def _store_data(self, lang: str):
        data = self._download_data(lang=lang)
        file_path = join(self.store, f'{lang}.jsonl')
        with open(file_path, 'w') as f:
            for item in data:
                print(json.dumps(item), file=f)

        with open(join(self.store, 'split'), 'w') as f:
            print(self.split, file=f)
        print(f'FLORES+ data for {lang} has been stored.')

    def _get_data(self, lang: str, num_of_sents: int = 300) -> list[dict[str, str]]:
        if not self._same_split():
            delete_files_in_folder(self.store)
            self._store_data(lang=lang)

        stored_langs = [f for f in os.listdir(
            self.store) if f.endswith('.jsonl')]

        if f'{lang}.jsonl' not in stored_langs:
            self._store_data(lang=lang)

        file_path = join(self.store, f'{lang}.jsonl')
        with open(file_path, 'r') as f:
            data = []
            for i, ln in enumerate(f):
                if i >= num_of_sents:
                    return data
                data.append(json.loads(ln))

    def _load_sents_for_lang(self, lang: str, num_of_sents: int = 300) -> list[str]:
        data = self._get_data(lang, num_of_sents=num_of_sents)
        sents = [o['text'] for o in data]
        return sents

    def get_sentences(self, *langs: tuple[str], num_of_sents: int = 300) -> dict[str, list[str]]:
        lang_sents = {}
        for lang in langs:
            assert lang in self.langs, 'Only the 11 European languages should be supported by the FloresManager'
            lang_sents[lang] = self._load_sents_for_lang(
                lang, num_of_sents=num_of_sents)
        return lang_sents

    def get_sentence_pairs(self, src_lang: str, tgt_lang: str, num_of_sents: int = 300) -> tuple[list[str], list[str]]:
        '''
        Returns sentences specified language pair
        '''
        out = self.get_sentences(src_lang, tgt_lang, num_of_sents=num_of_sents)
        return out[src_lang], out[tgt_lang]


class Opus100Manager(DataManager):
    # NOTE: OPUS100 is English-centric
    EURO_LANGS = {
        'de': 'de-en',
        'da': 'da-en',
        'el': 'el-en',
        'pt': 'en-pt',
        'sv': 'en-sv',
        'es': 'en-es',
        'fi': 'en-fi',
        'fr': 'en-fr',
        'it': 'en-it',
        'nl': 'en-nl'
    }

    def __init__(self, split: str = 'test', size: int = 500):
        self.store = get_env_variables('OPUS_100_STORE')
        self.langs = Opus100Manager.EURO_LANGS
        self.split = f'{split}[:{size}]'
        self.name = "Helsinki-NLP/opus-100"

    def _same_split(self) -> bool:
        split_file = join(self.store, 'split')
        try:
            with open(split_file, 'r') as f:
                split = f.readline()
        except FileNotFoundError:
            return False
        return split.strip() == self.split

    def _download_data(self, lang: str) -> Dataset:
        missing = self.langs[lang]
        print(f'Files for {lang} must be downloaded.')

        dataset = load_dataset(self.name,
                               missing, split=self.split)
        return dataset

    def _store_data(self, lang: str):
        data = self._download_data(lang=lang)
        if data is None:
            return
        file_path = join(self.store, f'{lang}.jsonl')
        with open(file_path, 'w') as f:
            for item in data:
                print(json.dumps(item['translation']), file=f)

        with open(join(self.store, 'split'), 'w') as f:
            print(self.split, file=f)
        print(f'OPUS-100 data for {lang} has been stored.')

    def _get_data(self, lang, num_of_sents: int = 300) -> list[dict[str, str]]:
        if not self._same_split():
            delete_files_in_folder(self.store)
            self._store_data(lang=lang)

        stored_langs = [f for f in os.listdir(
            self.store) if f.endswith('.jsonl')]

        if f'{lang}.jsonl' not in stored_langs:
            self._store_data(lang=lang)

        file_path = join(self.store, f'{lang}.jsonl')

        with open(file_path, 'r') as f:
            data = []
            for i, ln in enumerate(f):
                if i >= num_of_sents:
                    return data
                data.append(json.loads(ln))

    def get_sentence_pairs(self, src_lang: str, tgt_lang: str = 'en', num_of_sents: int = 300) -> tuple[list[str], list[str]]:
        '''
        Returns sentences of specified language-English pair
        '''
        check = [src_lang, tgt_lang]
        assert 'en' in check, 'This corpus only provides language pairs aligned to English, either src_lang or tgt_lang must be English!'

        if tgt_lang == 'en':
            data = self._get_data(src_lang, num_of_sents=num_of_sents)
            src_sents = [o[src_lang] for o in data]
            tgt_sents = [o['en'] for o in data]
        else:
            data = self._get_data(tgt_lang, num_of_sents=num_of_sents)
            tgt_sents = [o[tgt_lang] for o in data]
            src_sents = [o['en'] for o in data]
        return src_sents[:num_of_sents], tgt_sents[:num_of_sents]


class EPManager(DataManager):
    EURO_LANGS = set(['de', 'da', 'el', 'es',
                      'pt', 'nl', 'fi', 'sv', 'fr', 'it', 'en'])

    # NOTE: Different from FLORES+, 55 different language pairs, thus 55 different alignments, much more data
    EP_HELSINKI_PAIRS = set(['da-de',
                            'da-el',
                             'da-en',
                             'da-es',
                             'da-fi',
                             'da-fr',
                             'da-it',
                             'da-nl',
                             'da-pt',
                             'da-sv',
                             'de-el',
                             'de-en',
                             'de-es',
                             'de-fi',
                             'de-fr',
                             'de-it',
                             'de-nl',
                             'de-pt',
                             'de-sv',
                             'el-en',
                             'el-es',
                             'el-fi',
                             'el-fr',
                             'el-it',
                             'el-nl',
                             'el-pt',
                             'el-sv',
                             'en-es',
                             'en-fi',
                             'en-fr',
                             'en-it',
                             'en-nl',
                             'en-pt',
                             'en-sv',
                             'es-fi',
                             'es-fr',
                             'es-it',
                             'es-nl',
                             'es-pt',
                             'es-sv',
                             'fi-fr',
                             'fi-it',
                             'fi-nl',
                             'fi-pt',
                             'fi-sv',
                             'fr-it',
                             'fr-nl',
                             'fr-pt',
                             'fr-sv',
                             'it-nl',
                             'it-pt',
                             'it-sv',
                             'nl-pt',
                             'nl-sv',
                             'pt-sv'])

    def __init__(self, size: int = 500):
        self.pairs = EPManager.EP_HELSINKI_PAIRS
        self.langs = EPManager.EURO_LANGS
        self.store = get_env_variables('EUROPARL_STORE')
        # This dataset has only train split
        # Too avoid downloading everything, we shrink the size to the first 5%
        self.split = f'train[:{size}]'
        self.name = "Helsinki-NLP/europarl"

    def _same_split(self):
        split_file = join(self.store, 'split')
        try:
            with open(split_file, 'r') as f:
                split = f.readline()
        except FileNotFoundError:
            return False
        return split.strip() == self.split

    def _get_pair(self, lang1: str, lang2: str) -> str:
        assert lang1 in self.langs and lang2 in self.langs, 'Language pair not supported by corpus'
        pair1 = f'{lang1}-{lang2}'
        pair2 = f'{lang2}-{lang1}'
        assert pair1 in self.pairs or pair2 in self.pairs, 'Language pair not supported'
        if pair1 in self.pairs:
            return pair1
        if pair2 in self.pairs:
            return pair2

    def _download_data(self, pair: str) -> Dataset:
        print(f'Files for {pair} must be downloaded.')
        dataset = load_dataset(self.name, pair, split=self.split)
        return dataset['translation']

    def _store_data(self, pair: str):
        data = self._download_data(pair)
        file_path = join(self.store, f'{pair}.jsonl')
        with open(file_path, 'w') as f:
            for item in data:
                print(json.dumps(item), file=f)

        with open(join(self.store, 'split'), 'w') as f:
            print(self.split, file=f)
        print(f'EuroParl data for {pair} has been stored')

    def _get_data(self, pair: str, num_of_sents: int = 300) -> list[dict[str, str]]:
        if not self._same_split():
            delete_files_in_folder(self.store)
            self._store_data(pair)

        stored_langs = [f for f in os.listdir(
            self.store) if f.endswith('.jsonl')]

        if f'{pair}.jsonl' not in stored_langs:
            self._store_data(pair)

        file_path = join(self.store, f'{pair}.jsonl')
        with open(file_path, 'r') as f:
            data = []
            for i, ln in enumerate(f):
                if i >= num_of_sents:
                    return data
                data.append(json.loads(ln))
        return data

    def get_sentence_pairs(self, src_lang: str, tgt_lang: str, num_of_sents: int = 300) -> tuple[list[str], list[str]]:
        pair = self._get_pair(src_lang, tgt_lang)
        data = self._get_data(pair, num_of_sents=num_of_sents)
        src_sents = [o[src_lang] for o in data]
        tgt_sents = [o[tgt_lang] for o in data]
        return src_sents[:num_of_sents], tgt_sents[:num_of_sents]
