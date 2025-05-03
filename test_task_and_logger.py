from scripts.translators import MockClient
from scripts.task import TranslationTask
from scripts.logger import MyLogger, RetryLog
from scripts.data_management import Opus100Manager, EuroParlManager
from random import choice
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


def get_pairs(num):
    pairs = Opus100Manager.EURO_ISO_2_PAIR
    pairs = [tuple(pair.split('-')) for pair in pairs.values()]
    return pairs[:num]


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
    pairs = get_pairs(3)
    setup_teardown(test_folder, lambda: task_run(
        pairs=pairs, mt_folder=test_folder))


def test_task_with_error_and_retry():
    test_folder = 'tmp_test'
    pairs = get_pairs(4)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger, planned_errors=[
                     pairs[0], pairs[1], pairs[3]], planned_fails=[pairs[1], pairs[1], pairs[3]])

    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=2,
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    expected_logs = 4+6-1
    # with zero failure = 4 logs
    # with max_retries=2, max 4 * 3 = 12 logs
    assert len(log_data) == expected_logs
    expected_fail_logs = 6
    expected_error_logs = 3
    actual_fail_logs = 0
    actual_error_logs = 0
    for log in log_data:
        if 'failure' in log['verdict']:
            actual_fail_logs += 1
        if 'error' in log['verdict']:
            actual_error_logs += 1
            assert log['verdict']['error'] == 'MockError'
    assert actual_fail_logs == expected_fail_logs
    assert actual_error_logs == expected_error_logs


def test_logging_and_retry():
    test_folder = 'tmp_test'
    pairs = get_pairs(4)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
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


def test_logging_with_manual_retry():
    test_folder = 'tmp_test'
    pairs = get_pairs(4)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    cli = MockClient(logger=logger)
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    target_pair = [pairs[1], pairs[-1]]
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    log_ids = [log['translation']['id'] for log in log_data]
    target_ids = [log_ids[1], log_ids[-1]]
    retry = RetryLog(pairs=target_pair, log_ids=target_ids,
                     reasons=['Returned Src text', 'BLEU score single digit'])
    new_logger = MyLogger(logfile=logfile, retry=retry)
    cli = MockClient(logger=new_logger)
    task = TranslationTask(
        target_pairs=target_pair,
        dm=dm,
        client=cli,
        logger=new_logger,
        mt_folder=test_folder,
        num_of_sents=400,
        manual_retry=True,
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    expected_logs = len(pairs) + 2
    assert len(log_data) == expected_logs
    assert log_data[-1]['manual_retry']['reason'] == 'BLEU score single digit'
    assert log_data[-2]['manual_retry']['reason'] == 'Returned Src text'


def test_logging_with_scenario_1():
    test_folder = 'tmp_test'
    pairs = get_pairs(2)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    scenario = [0, 1, 2, 0]
    cli = MockClient(logger=logger, scenario=scenario)
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=2,
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    assert len(log_data) == len(scenario)
    assert log_data[0]['verdict']['success'] == 'Translation accepted'
    assert log_data[1]['verdict']['failure'] == 'Translation rejected'
    assert log_data[2]['verdict']['failure'] == 'Translation failed'
    assert log_data[3]['verdict']['success'] == 'Translation accepted'


def test_logging_with_scenario_2():
    test_folder = 'tmp_test'
    pairs = get_pairs(3)
    dm = Opus100Manager()
    logfile = StringIO()
    logger = MyLogger(logfile=logfile)
    scenario = [0, 1, 2, 2, 0, 1, 1, 2, 2]
    cli = MockClient(logger=logger, scenario=scenario)
    task = TranslationTask(
        target_pairs=pairs,
        dm=dm,
        client=cli,
        logger=logger,
        mt_folder=test_folder,
        num_of_sents=400,
        max_retries=3,
        retry_delay=0
    )
    setup_teardown(test_folder, lambda: silent_task_run(task.run))
    logvalues = logfile.getvalue()
    log_data = [json.loads(ln) for ln in logvalues.splitlines()]
    assert len(log_data) == len(scenario)
    for idx, s in enumerate(scenario):
        if s == 0:
            assert log_data[idx]['verdict']['success'] == 'Translation accepted'
        elif s == 1:
            assert log_data[idx]['verdict']['failure'] == 'Translation rejected'
        elif s == 2:
            assert log_data[idx]['verdict']['failure'] == 'Translation failed'
