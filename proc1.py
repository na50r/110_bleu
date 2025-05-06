from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config
from scripts.translators import GPTClient, DeeplClient
logging_config('proc1.log')


class Proc1(Procedure):
    def __init__(self):
        # Define all English including pairs
        en_pairs = Opus100Manager.get_pairs()
        
        # Define folder hierarchy of where translations should be stored
        main_folder = 'tasks'
        sub_folder = join(main_folder, 'proc1')

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager(), Opus100Manager()]
        self.dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in self.dm_ids]
        self.tasks = {dm_id: {} for dm_id in self.dm_ids}

        # Define logger and clients
        logger = TranslationLogger(logfile=join(main_folder, 'proc1.jsonl'))

        cli_gpt = GPTClient(logger=logger)
        cli_deepl = DeeplClient(logger=logger)
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
                    acceptable_range=(360, 480)
                )
                self.tasks[dm_id][client.model] = task
                num_of_tasks += 1
        assert num_of_tasks == 6


if __name__ == '__main__':
    desc = '''Procedure 1 Task Manager
    Compute translations from and into English for 10 European languages accross 3 Corpora and 2 Translators
    Implements 6 Tasks distinguished by Dataset and Translator
    Number of Input sentences: 400, within acceptable range of 360-480 sentences
    '''
    main(parser=proc_parser(desc=desc), proc=Proc1())
