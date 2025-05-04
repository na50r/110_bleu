from scripts.procedure import main, proc_parser, logging_config, Process
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger
from scripts.translators import GPTClient, DeeplClient
from random import sample
logging_config('procX.log')

class ProcX(Process):
    def __init__(self):
        # Define all English including pairs
        en_pairs = sample(Opus100Manager.get_pairs(), k=2)
        
        # Define folder hierarchy of where translations should be stored
        main_folder = 'tmp'
        sub_folder = join(main_folder, 'procX')

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager(), Opus100Manager()]
        self.dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in self.dm_ids]
        self.tasks = {dm_id: {} for dm_id in self.dm_ids}

        # Define the clients and logger
        logger = TranslationLogger(logfile=join(main_folder, 'procX.jsonl'))

        cli_gpt = GPTClient(logger=logger)
        clients = [cli_gpt]
        self.model_ids = [cli.model for cli in clients]

        # Create tasks
        for dm, folder, dm_id in zip(dms, dm_folders, self.dm_ids):
            for client in clients:
                task = TranslationTask(
                    target_pairs=[en_pairs[0], en_pairs[1]],
                    dm=dm,
                    client=client,
                    logger=logger,
                    mt_folder=join(folder, client.model),
                    num_of_sents=10,
                    acceptable_range=(5, 15),
                    retry_delay=0
                )
                self.tasks[dm_id][client.model] = task

if __name__ == '__main__':
    desc = 'Procedure X Task Manager (For Testing Purposes)'
    main(parser=proc_parser(desc=desc), proc=ProcX())
