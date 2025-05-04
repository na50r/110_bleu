from scripts.translators import MockClient
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, RetryLog
from scripts.data_management import Opus100Manager, EuroParlManager
from random import choice, sample
from io import StringIO
import os
import shutil
import json
import sys


def silent_task_run(func):
    # Hide stdout
    # Based on: https://stackoverflow.com/a/2829036
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    func()
    sys.stdout = original_stdout


def get_sample_pairs(dm, k=2):
    pairs = dm.get_pairs()
    pairs = sample(pairs, k=k)
    return pairs


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


def task_run(mt_folder):
    # Generic, random task run on 400 sentences
    dms = [EuroParlManager, Opus100Manager]
    dm = choice(dms)
    pairs = get_sample_pairs(dm)
    dm = dm()
    logfile = StringIO()
    logger = TranslationLogger(logfile=logfile)
    cli = MockClient(logger=logger)
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=mt_folder,
        num_of_sents=400,
    )
    silent_task_run(task.run)


def test_task():
    test_folder = 'tmp_test'
    setup_teardown(test_folder, lambda: task_run(mt_folder=test_folder))


def test_retry_and_fail_files():
    test_folder = 'tmp_test'
    pairs = get_sample_pairs(Opus100Manager, k=4)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = TranslationLogger(logfile=logfile)
    cli = MockClient(logger=logger, planned_fails=[
                     pairs[1], pairs[1], pairs[3]])

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
    fail_files = [f for f in files if f.endswith(
        '_fail1.txt') or f.endswith('_fail2.txt')]
    assert len(fail_files) == 3
    assert f'{pairs[1][0]}-{pairs[1][1]}.txt' not in files
    teardown(test_folder)


def test_meta_data_in_log():
    test_folder = 'tmp_test'
    pairs = get_sample_pairs(Opus100Manager, k=4)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = TranslationLogger(logfile=logfile)
    cli = MockClient(logger=logger, model='gpt-4.1-2077-01-01')

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
    log_data = [json.loads(ln) for ln in logfile.getvalue().splitlines()]
    assert len(log_data) == 4
    assert [log['verdict'] for log in log_data] == ['accepted'] * 4
    assert [log['src_lang'] for log in log_data] == [p[0] for p in pairs]
    assert [log['tgt_lang'] for log in log_data] == [p[1] for p in pairs]
    assert [log['id'] for log in log_data] == [
        f'{task.id}-{i:04d}' for i in range(1, 5)]
    assert [log['translator']
            for log in log_data] == ['gpt-4.1-2077-01-01'] * 4
    assert [log['dataset'] for log in log_data] == [dm.name] * 4


SCENARIOS = {
    'A': {
        'dm': Opus100Manager,
        'pairs': get_sample_pairs(Opus100Manager, k=4),
        'scenario': [0, 1, 1, 0, 0, 2, 2, 2],
        'logs': 5,
        'verdicts': ['accepted', 'rejected', 'rejected', 'accepted', 'accepted'],
        'max_retries': 2
    },
    'B': {
        'dm': EuroParlManager,
        'pairs': get_sample_pairs(EuroParlManager, k=4),
        'scenario': [2, 2, 2, 0, 2, 1, 0, 1, 2, 2],
        'logs': 4,
        'verdicts': ['accepted', 'rejected', 'accepted', 'rejected'],
        'max_retries': 2
    }
}


def test_logging_with_scenario_A():
    test_folder = 'tmp_test'
    dm = SCENARIOS['A']['dm']()
    pairs = SCENARIOS['A']['pairs']
    logfile = StringIO()
    logger = TranslationLogger(logfile=logfile)
    cli = MockClient(logger=logger, scenario=SCENARIOS['A']['scenario'])

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=SCENARIOS['A']['max_retries'],
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    log_data = [json.loads(ln) for ln in logfile.getvalue().splitlines()]
    assert len(log_data) == SCENARIOS['A']['logs']
    assert [log['verdict'] for log in log_data] == SCENARIOS['A']['verdicts']


def test_logging_with_scenario_B():
    test_folder = 'tmp_test'
    dm = SCENARIOS['B']['dm']()
    pairs = SCENARIOS['B']['pairs']
    logfile = StringIO()
    logger = TranslationLogger(logfile=logfile)
    cli = MockClient(logger=logger, scenario=SCENARIOS['B']['scenario'])
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=SCENARIOS['B']['max_retries'],
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    log_data = [json.loads(ln) for ln in logfile.getvalue().splitlines()]
    assert len(log_data) == SCENARIOS['B']['logs']
    assert [log['verdict'] for log in log_data] == SCENARIOS['B']['verdicts']
