from scripts.util import get_git_revision_short_hash, split_sents
import time
import json
import tiktoken
from datetime import datetime
from typing import TextIO, Any
import uuid


class ReRun:
    def __init__(self, pairs: list[tuple[str, str]] = [], log_ids: list[str] = [], reasons: list[str] = []):
        self.items = {pair: {'log_id': log_id, 'reason': reason}
                      for pair, log_id, reason in zip(pairs, log_ids, reasons)}
        

    def get_log_id(self, pair: tuple[str, str]):
        return self.items[pair]['log_id']

    def get_reason(self, pair: tuple[str, str]):
        return self.items[pair]['reason']
    

class MyLogger:
    def __init__(self, logfile: str | TextIO, rerun: ReRun = ReRun()):
        self.logfile = logfile
        self.is_path = isinstance(logfile, str)
        self.log = {'git_hash': get_git_revision_short_hash()}
        self.rerun = rerun

    def add_entry(self, **kwargs):
        self.log.update(kwargs)

    def add_dataset_info(self, name: str, num_of_sents: int, start_idx: int = 0, **kwargs):
        dataset_log = {
            'name': name,
            'num_of_sents': num_of_sents,
            'start_idx': start_idx
        }
        dataset_log.update(kwargs)
        self.add_entry(dataset=dataset_log)

    def add_rerun_info(self, pair: tuple[str, str]):
        rerun_log = {
            'log_id': self.rerun.get_log_id(pair),
            'reason': self.rerun.get_reason(pair)
        }
        self.add_entry(rerun=rerun_log)

    def start(self, src_lang: str, tgt_lang: str, src_text: str, translator: str):
        self.current = TranslationLog(src_lang, tgt_lang, src_text, translator)
        return self.current

    def finish(self, tgt_text: str, **kwargs):
        if self.current:
            self.current.finish(tgt_text, **kwargs)
            self.log['translation'] = self.current.to_dict()
            self._write_log(self.log)
            del self.current

    def _write_log(self, log_dict: dict[str, Any]):
        if self.is_path:
            with open(self.logfile, 'a') as f:
                print(json.dumps(log_dict), file=f)
        else:
            print(json.dumps(log_dict), file=self.logfile)

    def log_error(self, error: Exception, src_lang: str, tgt_lang: str, translator: str):
        log = self.start(src_lang, tgt_lang, src_text='',
                         translator=translator)
        log.error_msg = f"Translation {src_lang} to {tgt_lang} failed"
        log.error = str(error)
        self.log['translation'] = log.to_dict()
        self._write_log(self.log)
        del self.current


class TranslationLog:
    def __init__(self, src_lang: str, tgt_lang: str, src_text: str, translator: str, tokenizer: str = 'gpt-4o'):
        self.enc = tiktoken.encoding_for_model(tokenizer)

        self.translator = translator
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.start = time.time()
        self.id = str(uuid.uuid4())
        self.in_lines = len(src_text.splitlines())
        self.in_sents = len(split_sents(src_text, lang=src_lang))
        self.start_timestamp = str(datetime.now().astimezone())
        self.in_chars = len(src_text)
        self.in_tokens = len(self.enc.encode(src_text))

    def finish(self, tgt_text: str, **kwargs):
        self.end = time.time()
        self.end_timestamp = str(datetime.now().astimezone())
        self.time = self.end - self.start
        self.out_chars = len(tgt_text)
        self.out_lines = len(tgt_text.splitlines())
        self.out_sents = len(split_sents(tgt_text, lang=self.tgt_lang))
        self.out_tokens = len(self.enc.encode(tgt_text))

        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        out = vars(self)
        del out['enc']
        return out
