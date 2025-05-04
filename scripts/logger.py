from scripts.util import split_sents
import time
import json
import tiktoken
from typing import TextIO

class RetryLog:
    def __init__(self, pairs: list[tuple[str, str]] = [], log_ids: list[str] = [], reasons: list[str] = []):
        self.items = {pair: {'log_id': log_id, 'reason': reason}
                      for pair, log_id, reason in zip(pairs, log_ids, reasons)}

    def get_log_id(self, pair: tuple[str, str]):
        return self.items[pair]['log_id']

    def get_reason(self, pair: tuple[str, str]):
        return self.items[pair]['reason']


class TranslationLogger:
    def __init__(self, logfile: str | TextIO, tokenizer: str = 'gpt-4o', retry: RetryLog = RetryLog()):
        self.logfile = logfile
        self.is_path = isinstance(logfile, str)
        self.retry = retry
        self.current = None
        self.enc = tiktoken.encoding_for_model(tokenizer)
        
    def add_entry(self, **kwargs):
        if self.current is None:
            return
        self.current.update(kwargs)

    def add_manual_retry_info(self, pair: tuple[str, str]):
        retry_log = {
            'prev_id': self.retry.get_log_id(pair),
            'reason': self.retry.get_reason(pair)
        }
        self.add_entry(manual_retry=retry_log)

    def start(self, src_lang: str, tgt_lang: str, src_text: str):
        log = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'start': time.time(),
            'in_lines': len(src_text.splitlines()),
            'in_sents' : len(split_sents(src_text, lang=src_lang)),
            'in_chars':  len(src_text),
            'in_tokens': len(self.enc.encode(src_text)),
        }
        self.current = log
        return self.current

    def finish(self, tgt_text: str, **kwargs):
        if self.current is not None:
            self.current['end'] = time.time()
            self.current['out_chars'] = len(tgt_text)
            self.current['out_lines'] = len(tgt_text.splitlines())
            self.current['out_sents'] = len(split_sents(tgt_text, lang=self.current['tgt_lang']))
            self.current['out_tokens'] = len(self.enc.encode(tgt_text))
            self.current.update(kwargs)

    def write_log(self, verdict: bool = True):
        if self.current is None:
            return
        if verdict:
            self.current['verdict'] = 'accepted'
        else:
            self.current['verdict'] = 'rejected'       
        if self.is_path:
            with open(self.logfile, 'a') as f:
                print(json.dumps(self.current), file=f)
        else:
            print(json.dumps(self.current), file=self.logfile)
        self.current = None

