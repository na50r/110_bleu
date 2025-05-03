from process import main, proc_parser, Process
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import MyLogger
from test_task_and_logger import MockClient
import sys
import logging

# Based on https://stackoverflow.com/a/40909549
level = logging.INFO
format = '  %(message)s'
handlers = [logging.FileHandler(join('tmp', 'phase0.log')), logging.StreamHandler()]
logging.basicConfig(level=level, format=format, handlers=handlers)


class Phase0(Process):
    def __init__(self):
        # Define all English including pairs
        langs = Opus100Manager.EURO_ISO_2_PAIR.keys()
        possible = [tuple((lang, 'en')) for lang in sorted(langs)]
        extended = [(pair[1], pair[0]) for pair in possible]
        possible.extend(extended)
        en_pairs = possible

        # Define folder hierarchy of where translations should be stored
        main_folder = 'tmp'
        sub_folder = join(main_folder, 'phase0')

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager(), Opus100Manager()]
        self.dm_ids = [dm.name.split('/')[-1] for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in self.dm_ids]
        self.tasks = {dm_id: {} for dm_id in self.dm_ids}

        # Define the clients and logger
        logger = MyLogger(logfile=join(main_folder, 'phase0.jsonl'))

        cli_gpt = MockClient(logger=logger, model='gpt', planned_errors=[('de', 'en')], planned_fails=[('de', 'en')])
        cli_deepl = MockClient(logger=logger, model='deepl')
        clients = [cli_gpt, cli_deepl]
        self.model_ids = [cli.model for cli in clients]

        # Check if we have 20 pairs
        assert len(en_pairs) == 20
        
        # Create tasks
        num_of_tasks = 0
        for dm, folder, dm_id in zip(dms, dm_folders, self.dm_ids):
            for client in clients:
                task = TranslationTask(
                    target_pairs=en_pairs,
                    dm=dm,
                    client=client,
                    logger=logger,
                    mt_folder=join(folder, client.model),
                    num_of_sents=400,
                    acceptable_range=(360, 480),
                    retry_delay=0
                )
                self.tasks[dm_id][client.model] = task
                num_of_tasks += 1
        assert num_of_tasks == 6


if __name__ == '__main__':
    desc = 'Phase 0 Task Manager (Testing Purposes)'
    main(parser=proc_parser(desc=desc), proc=Phase0())
