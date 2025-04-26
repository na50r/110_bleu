from scripts.translators import TranslationClient
from scripts.task import TranslationTask
from scripts.util import MyLogger
from scripts.data_management import Opus100Manager, FloresPlusManager, EuroParlManager
from string import ascii_letters, ascii_uppercase, ascii_lowercase
from random import random, choice
from io import StringIO
import os
import shutil


class MockClient(TranslationClient):
    def __init__(self, logger=None, error_rate=0):
        self.client = None
        self.logger = logger
        self.model = 'mock'
        self.error_rate = error_rate

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
        in_text = '\n'.join(text)

        if self.logger:
            self.logger.start(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=in_text,
                translator=self.model,
            )
        out_text = self.encrypt(in_text)

        if self.logger:
            self.logger.finish(tgt_text=out_text)
        return out_text.splitlines()


def task_run(pairs, mt_folder, error_rate=0):
    dms = [FloresPlusManager, EuroParlManager, Opus100Manager]
    dm = choice(dms)()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, error_rate=error_rate)

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=mt_folder,
        num_of_sents=10
    )
    task.run()
    logvalues = logfile.getvalue()
    print(logvalues)


def test_run_1():
    test_folder = 'tmp_test'
    pairs = [('de', 'en')]
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    try:
        task_run(pairs=pairs, mt_folder=test_folder)
    finally:
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)


def test_run_2():
    test_folder = 'tmp_test'
    pairs = [('de', 'en'), ('en', 'de'), ('fr', 'en'), ('en', 'fr')]
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    try:
        task_run(pairs=pairs, mt_folder=test_folder, error_rate=0.5)
    finally:
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
