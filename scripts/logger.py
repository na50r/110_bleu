from scripts.util import split_sents
import time
import json
import tiktoken
from typing import TextIO
import logging

# Based on https://stackoverflow.com/a/40909549
# Adjusted with the help of ChatGPT due to unfamiliarity with logging
def logging_config(logfile='tmp.log'):
    fmt = '%(levelname)s: %(asctime)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt))

    logging.basicConfig(level=logging.DEBUG, handlers=[
                        file_handler, console_handler])


class RetryLog:
    def __init__(self, pairs: list[tuple[str, str]] = [], log_ids: list[str] = [], reasons: list[str] = []):
        self.items = {pair: {'log_id': log_id, 'reason': reason}
                      for pair, log_id, reason in zip(pairs, log_ids, reasons)}

    def get_log_id(self, pair: tuple[str, str]):
        return self.items[pair]['log_id']

    def get_reason(self, pair: tuple[str, str]):
        return self.items[pair]['reason']


class TranslationLogger:
    def __init__(self, logfile: str | TextIO, tokenizer: str = 'gpt-4o', retry_log: RetryLog = RetryLog()):
        self.logfile = logfile
        self.is_path = isinstance(logfile, str)
        self.retry_log = retry_log
        self.current_log = None
        self.enc = tiktoken.encoding_for_model(tokenizer)
        
    def add_entry(self, **kwargs):
        if self.current_log is None:
            return
        self.current_log.update(kwargs)

    def add_manual_retry_info(self, pair: tuple[str, str]):
        retry_log = {
            'prev_id': self.retry_log.get_log_id(pair),
            'reason': self.retry_log.get_reason(pair)
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
        self.current_log = log
        return self.current_log

    def finish(self, tgt_text: str, **kwargs):
        if self.current_log is not None:
            self.current_log['end'] = time.time()
            self.current_log['out_chars'] = len(tgt_text)
            self.current_log['out_lines'] = len(tgt_text.splitlines())
            self.current_log['out_sents'] = len(split_sents(tgt_text, lang=self.current_log['tgt_lang']))
            self.current_log['out_tokens'] = len(self.enc.encode(tgt_text))
            self.current_log.update(kwargs)

    def write_log(self, verdict: bool = True):
        if self.current_log is None:
            return
        if verdict:
            self.current_log['verdict'] = 'accepted'
        else:
            self.current_log['verdict'] = 'rejected'       
        if self.is_path:
            with open(self.logfile, 'a') as f:
                print(json.dumps(self.current_log), file=f)
        else:
            print(json.dumps(self.current_log), file=self.logfile)
        self.current_log = None

