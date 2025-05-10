from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config, RetryLog
from scripts.translators import GPTClient
logging_config('proc5.log')


class Proc5(Procedure):
    def __init__(self):
        main_folder = 'tasks'
        sub_folder = join(main_folder, 'proc5')
        logfile = join(main_folder, 'proc5.jsonl')
        # proc4, de-fi log id
        log_ids_flores = ['ee3252e9-ee96-4559-854b-f2ac3d0e912c-0005']
        reasons_flores = ['no accepted translation (output) yet (too large)']

        # Define retry task of FLORES+ dataset
        pairs_flores = [('de', 'fi')]
        retry_flores = RetryLog(
            pairs=pairs_flores, log_ids=log_ids_flores, reasons=reasons_flores)
        logger_flores = TranslationLogger(
            logfile=logfile, retry_log=retry_flores)
        cli_flores = GPTClient(logger=logger_flores, stream=True)

        task_flores = TranslationTask(
            target_pairs=pairs_flores,
            dm=FloresPlusManager(),
            client=cli_flores,
            logger=logger_flores,
            mt_folder=join(sub_folder, join('flores_plus', cli_flores.model)),
            num_of_sents=400,
            acceptable_range=(360, 480),
            manual_retry=True,
            max_retries=15,
        )

        # Define retry task of EuroParl dataset
        pairs_europarl = [('fi', 'el'), ('it', 'el'),
                          ('it', 'fi'), ('sv', 'el')]
        log_ids_europarl = [None, None, None, None]
        ep_reasons = ['Gateway 504 timeout, exceeded retries'] * 4
        retry_europarl = RetryLog(
            pairs=pairs_europarl, log_ids=log_ids_europarl, reasons=ep_reasons)
        logger_europarl = TranslationLogger(
            logfile=logfile, retry_log=retry_europarl)
        cli_europarl = GPTClient(logger=logger_europarl, stream=True)
        task_europarl = TranslationTask(
            target_pairs=pairs_europarl,
            dm=EuroParlManager(),
            client=cli_europarl,
            logger=logger_europarl,
            mt_folder=join(sub_folder, join('europarl', cli_europarl.model)),
            num_of_sents=400,
            acceptable_range=(360, 480),
            manual_retry=True,
            max_retries=5,
        )

        self.tasks = {
            'flores_plus': {
                cli_flores.model: task_flores
            },
            'europarl': {
                cli_europarl.model: task_europarl
            }
        }
        self.dm_ids = list(self.tasks.keys())
        self.model_ids = ['gpt-4.1-2025-04-14']


if __name__ == '__main__':
    desc = '''Procedure 5 Task Manager
    Retrying pairs that failed in Procedure 2, 3 and 4
    '''
    main(parser=proc_parser(desc=desc), proc=Proc5())
