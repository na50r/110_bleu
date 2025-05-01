from scripts.translators import TranslationClient
from scripts.task import TranslationTask
from scripts.logger import MyLogger, Retry
from scripts.data_management import Opus100Manager, EuroParlManager
from string import ascii_letters, ascii_uppercase, ascii_lowercase
from random import choice
from io import StringIO
import os
import shutil
import json
import sys


class MockClient(TranslationClient):
    def __init__(self, logger=None, planned_fails=[], planned_errors=[]):
        self.client = None
        self.logger = logger
        self.model = 'mock'
        self.planned_fails = planned_fails
        self.planned_errors = planned_errors

    def encrypt(self, text, key=13, direction=1, error_pair=None):
        if error_pair in self.planned_errors:
            self.planned_errors.remove(error_pair)
            raise (Exception(f'MockError'))

        # Code from: https://stackoverflow.com/a/34734063
        if direction == -1:
            key = 26 - key

        trans = str.maketrans(
            ascii_letters, ascii_lowercase[key:] + ascii_lowercase[:key] + ascii_uppercase[key:] + ascii_uppercase[:key])
        return text.translate(trans)

    def translate_document(self, text, src_lang, tgt_lang):
        in_text = '\n'.join(text)

        if self.logger:
            self.logger.start(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=in_text,
                translator=self.model,
            )
        out_text = self.encrypt(in_text, error_pair=(src_lang, tgt_lang))

        if (src_lang, tgt_lang) in self.planned_fails:
            tmp = out_text.splitlines()
            tmp = tmp[:round(len(tmp)/2)]
            out_text = '\n'.join(tmp)
            self.planned_fails.remove((src_lang, tgt_lang))

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



def setup(foldername):
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

def teardown(foldername):
    if os.path.exists(foldername):
        shutil.rmtree(foldername)

def setup_teardown(foldername, func):
    setup(foldername)
    try:
        func()
    finally:
        teardown(foldername)

def task_run(pairs, mt_folder):
    dms = [EuroParlManager, Opus100Manager]
    dm = choice(dms)()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger)

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=mt_folder,
        num_of_sents=50,
    )
    silent_task_run(task.run)


def test_task():
    test_folder = 'tmp_test'
    pairs = [('de', 'en')]
    setup_teardown(test_folder, lambda: task_run(
        pairs=pairs, mt_folder=test_folder))


def test_task_with_error_and_retry():
    test_folder = 'tmp_test'
    pairs = [('de', 'en'), ('en', 'de'), ('fr', 'en'), ('en', 'fr')]
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, planned_errors=[pairs[0], pairs[3]])
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=1,
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    expected_logs = len(pairs) + 2
    assert len(log_data) == expected_logs

def test_logging_and_retry():
    test_folder = 'tmp_test'
    pairs = [('de', 'en'), ('en', 'de'), ('fr', 'en'), ('en', 'fr')]
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, planned_fails=[pairs[1], pairs[1], pairs[3]])

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=1,
        retry_delay=0
    )

    setup(test_folder)
    silent_task_run(task.run)
    files = os.listdir(test_folder)
    fail_files = [f for f in files if f.endswith('_fail1.txt') or f.endswith('_fail2.txt')]
    assert len(fail_files) == 3
    assert f'{pairs[1][0]}-{pairs[1][1]}.txt' not in files
    teardown(test_folder)
    
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    expected_logs = len(pairs) + 2
    assert len(log_data) == expected_logs
    
    expected_fail_logs = 3
    actual_fail_logs = 0
    for log in log_data:
        if 'failure' in log['verdict']:
            actual_fail_logs += 1
    assert actual_fail_logs == expected_fail_logs

