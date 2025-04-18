import tiktoken
from datetime import datetime
import time
import json
import os
import uuid
from os.path import join, exists, isfile
from sentence_splitter import SentenceSplitter
from dotenv import load_dotenv
load_dotenv()

# Based on https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/


def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = join(folder_path, filename)
        if isfile(file_path):
            os.remove(file_path)

# Utitlity file that has several function useful for different cases
# Should be data independent but expects only European languages for now
# Adheres to fixed JSON format for alignments


def get_env_variables(*args):
    # Avoid caching in Jupyter Notebook
    for arg in args:
        os.environ.pop(arg, None)
    load_dotenv()
    if len(args) == 1:
        return os.getenv(args[0])

    return (os.getenv(arg) for arg in args)


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


def split_sents(text, lang):
    if lang in LANG_ISO:
        splitter = SentenceSplitter(language=lang)
        sents = splitter.split(text=text)
        sents = [sent.strip() for sent in sents]
        return sents
    else:
        raise Exception(f'The language {LANG_ISO[lang]} is not suppored yet.')


def store_sents(sents, folder_path, src_lang, tgt_lang):
    filename = f'{src_lang}-{tgt_lang}.txt'
    if not exists(folder_path):
        os.makedirs(folder_path)
    with open(join(folder_path, filename), 'w') as f:
        for sent in sents:
            print(sent.strip(), file=f)


def load_sents(folder_path, src_lang, tgt_lang):
    filename = join(folder_path, f'{src_lang}-{tgt_lang}.txt')
    with open(filename, 'r') as f:
        data = [s.strip() for s in f.readlines()]
    return data


def triplet_align_sents(mt_sents: list[str], tgt_sents: list[str], src_sents: list[str], src_lang: str, tgt_lang: str, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(join(folder_path, f'{src_lang}-{tgt_lang}.jsonl'), 'a') as f:
        for mt, ref, src in zip(mt_sents, tgt_sents, src_sents):
            obj = dict()
            obj['mt'] = mt
            obj['ref'] = ref
            obj['src'] = src
            print(json.dumps(obj), file=f)


def align_src_mt_sents(src_sents, mt_sents, src_lang, mt_lang, folder_path):
    if not exists(folder_path):
        os.makedirs(folder_path)

    mt_filename = f'{src_lang}-{mt_lang}.{mt_lang}'
    src_filename = f'{src_lang}-{mt_lang}.{src_lang}'

    mt_file = join(folder_path, mt_filename)
    src_file = join(folder_path, src_filename)
    if exists(mt_file) and exists(src_file):
        print(f'{src_lang} and {mt_lang} already aligned!')
        return

    from bertalign import Bertalign
    mt_text = '\n'.join(mt_sents)
    src_text = '\n'.join(src_sents)
    aligner = Bertalign(
        src=src_text,
        tgt=mt_text,
        src_lang=src_lang,
        tgt_lang=mt_lang,
        model='paraphrase-multilingual-MiniLM-L12-v2'
    )
    aligner.align_sents()
    src_sents_a, mt_sents_a = aligner.get_sents()

    with open(join(folder_path, src_filename), 'w') as f:
        for sent in src_sents_a:
            print(sent, file=f)

    with open(join(folder_path, mt_filename), 'w') as f:
        for sent in mt_sents_a:
            print(sent, file=f)

    return src_sents_a, mt_sents_a


def post_triplet_alignment(src_sents_org, src_sents_ali, ref_sents_org, mt_sents_ali, src_lang, ref_lang, folder_path):
    if not exists(folder_path):
        os.makedirs(folder_path)

    aligned_cnt = 0
    out_file = f'{src_lang}-{ref_lang}.jsonl'
    with open(join(folder_path, out_file), 'w') as f:
        for x, sent in enumerate(src_sents_org):
            for y, src_sent in enumerate(src_sents_ali):
                if sent.strip() == src_sent.strip():
                    obj = dict()
                    obj['src'] = sent.strip()
                    obj['ref'] = ref_sents_org[x].strip()
                    obj['mt'] = mt_sents_ali[y].strip()
                    if obj['src'] != '' and obj['ref'] != '' and obj['mt'] != '':
                        print(json.dumps(obj), file=f)
                        aligned_cnt += 1
                else:
                    continue
    print(f"{aligned_cnt} sents aligned for {src_lang} and {ref_lang}")


class MyLogger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.dataset = None
        self.current = None

    def add_dataset_info(self, name, num_of_sents, start_idx=0, **kwargs):
        self.dataset = {
            'name': name,
            'num_of_sents': num_of_sents,
            'start_idx': start_idx
        }
        self.dataset.update(kwargs)

    def start(self, src_lang, tgt_lang, src_text, translator):
        self.current = Log(src_lang, tgt_lang, src_text,
                           translator, dataset=self.dataset)
        return self.current

    def finish(self, tgt_text, out_tokens=None, in_tokens=None):
        if self.current:
            self.current.finish(tgt_text, out_tokens, in_tokens)
            self._write_log(self.current.to_dict())
            self.current = None

    def _write_log(self, log_dict):
        with open(self.logfile, 'a') as f:
            print(json.dumps(log_dict), file=f)

    def log_error(self, error, src_lang, tgt_lang, translator):
        log = self.start(src_lang, tgt_lang, src_text=None,
                         translator=translator)
        log.error_msg = f"Translation {src_lang} to {tgt_lang} failed"
        log.error = str(error)
        self._write_log(log.to_dict())
        self.current = None


class Log:
    def __init__(self, src_lang, tgt_lang, src_text, translator, dataset):
        self.enc = tiktoken.encoding_for_model("gpt-4o")

        self.translator = translator
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.start = time.time()
        self.id = str(uuid.uuid4())
        self.in_lines = len(src_text.splitlines()) if src_text else None
        self.in_sents = len(split_sents(src_text, lang=src_lang)
                            ) if src_text else None
        self.stamp = str(datetime.now().astimezone())
        self.in_chars = len(src_text) if src_text else None
        self.in_tiktoks = len(self.enc.encode(src_text)) if src_text else None
        self.dataset = dataset

        self.out_chars = None
        self.out_tiktoks = None
        self.out_sents = None
        self.in_toks = None
        self.out_toks = None
        self.out_lines = None
        self.end = None
        self.error = None
        self.error_msg = None

    def finish(self, tgt_text, out_tokens=None, in_tokens=None):
        self.end = time.time()
        self.time = self.end - self.start
        self.out_chars = len(tgt_text)
        self.out_lines = len(tgt_text.splitlines())
        self.out_sents = len(split_sents(tgt_text, lang=self.tgt_lang))
        self.out_tiktoks = len(self.enc.encode(tgt_text))
        self.out_toks = out_tokens
        self.in_toks = in_tokens

    def to_dict(self):
        out = vars(self)
        del out['enc']
        return out
