from scripts.translators import TranslationClient
from scripts.task import TranslationTask
from scripts.logger import MyLogger, Retry
from scripts.data_management import Opus100Manager, FloresPlusManager, EuroParlManager
from string import ascii_letters, ascii_uppercase, ascii_lowercase
from random import random, choice
from io import StringIO
import os
import shutil
import json
import sys


class MockClient(TranslationClient):
    def __init__(self, logger=None, error_rate=0, planned_fails=[]):
        self.client = None
        self.logger = logger
        self.model = 'mock'
        self.error_rate = error_rate
        self.planned_fails = planned_fails
        self.translation_cnt = -1

    def encrypt(self, text, key=13, direction=1):
        if self.error_rate > 0 and random() < self.error_rate:
            raise (Exception(f'MockError'))

        # Code from: https://stackoverflow.com/a/34734063
        if direction == -1:
            key = 26 - key

        trans = str.maketrans(
            ascii_letters, ascii_lowercase[key:] + ascii_lowercase[:key] + ascii_uppercase[key:] + ascii_uppercase[:key])
        return text.translate(trans)

    def translate_document(self, text, src_lang, tgt_lang):
        self.translation_cnt += 1
        in_text = '\n'.join(text)

        if self.logger:
            self.logger.start(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=in_text,
                translator=self.model,
            )
        out_text = self.encrypt(in_text)

        if self.translation_cnt in self.planned_fails:
            tmp = out_text.splitlines()
            tmp = tmp[:round(len(tmp)/2)]
            out_text = '\n'.join(tmp)

        if self.logger:
            self.logger.finish(tgt_text=out_text)
        return out_text.splitlines()


def silent_task_run(func):
    # Hide stdout
    # Based on: https://stackoverflow.com/a/2829036
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    func()
    sys.stdout = original_stdout


def setup_teardown(foldername, func):
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)
    try:
        func()
    finally:
        if os.path.exists(foldername):
            shutil.rmtree(foldername)


def task_run(pairs, mt_folder, error_rate=0, planned_fails=[]):
    dms = [FloresPlusManager, EuroParlManager, Opus100Manager]
    dm = choice(dms)()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, error_rate=error_rate,
                     planned_fails=planned_fails)

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=mt_folder,
        num_of_sents=50
    )

    silent_task_run(task.run)

    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    rest = []
    for log in log_data:
        if log['translation'].get('error', None) != None:
            rest.append(log)
    if len(rest) != 0:
        for log in rest:
            assert log['translation']['error'] == 'MockError'


def test_task():
    test_folder = 'tmp_test'
    pairs = [('de', 'en')]
    setup_teardown(test_folder, lambda: task_run(
        pairs=pairs, mt_folder=test_folder))


def test_task_with_error():
    test_folder = 'tmp_test'
    pairs = [('de', 'en'), ('en', 'de'), ('fr', 'en'), ('en', 'fr')]
    setup_teardown(test_folder, lambda: task_run(
        pairs=pairs, mt_folder=test_folder))


def test_logging_with_retry():
    test_folder = 'tmp_test'
    pairs = [('de', 'en'), ('en', 'de'), ('fr', 'en'), ('en', 'fr')]
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, planned_fails=[1, 3])

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400
    )

    setup_teardown(test_folder, lambda: silent_task_run(task.run))

    logvalues = logfile.getvalue()
    log_ids = []
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    for log in log_data:
        src_lang = log['translation']['src_lang']
        tgt_lang = log['translation']['tgt_lang']
        if (src_lang == 'en' and tgt_lang == 'de') or (src_lang == 'en' and tgt_lang == 'fr'):
            assert log['translation']['in_lines'] == 400 and log['translation']['out_lines'] == 200
            log_ids.append(log['translation']['id'])

    target_pairs = [('en', 'de'), ('en', 'fr')]
    retry = Retry(pairs=target_pairs, log_ids=log_ids,
                  reasons=['test1', 'test2'])
    new_logger = MyLogger(logfile=logfile, retry=retry)
    cli = MockClient(logger=new_logger)

    task = TranslationTask(
        target_pairs=target_pairs,
        dm=dm,
        client=cli,
        logger=new_logger,
        mt_folder=test_folder,
        num_of_sents=400,
        is_retry=True
    )

    setup_teardown(test_folder, lambda: silent_task_run(task.run))

    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    interest = log_data[-2:]
    assert interest[0]['retry']['reason'] == 'test1' and interest[1]['retry']['reason'] == 'test2'
    assert interest[0]['retry']['log_id'] == log_ids[0] and interest[1]['retry']['log_id'] == log_ids[1]
