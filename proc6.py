from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config
from scripts.translators import DeeplClient
logging_config('proc6.log')

class Proc6(Procedure):
    def __init__(self):
        # Define all English including pairs
        en_pairs = Opus100Manager.get_pairs()
        all_pairs = EuroParlManager.get_pairs()
        non_en_pairs = set(all_pairs) - set(en_pairs)

        langs = sorted(Opus100Manager.EURO_ISO_2_PAIR.keys())
        pairs = []
        for lang in langs:
            pairs.extend([p for p in non_en_pairs if p[0]==lang])
        
        assert len(pairs) == 90
        
        # Define folder hierarchy of where translations should be stored
        main_folder = 'tasks'
        sub_folder = join(main_folder, 'proc6')

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager()]
        self.dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in self.dm_ids]
        self.tasks = {dm_id: {} for dm_id in self.dm_ids}

        # Define logger and clients
        logger = TranslationLogger(logfile=join(main_folder, 'proc6.jsonl'))
        cli_deepl = DeeplClient(logger=logger)
        self.model_ids = [cli_deepl.model]
        
        for dm, folder, dm_id in zip(dms, dm_folders, self.dm_ids):
            task = TranslationTask(
                target_pairs=pairs,
                dm=dm,
                client=cli_deepl,
                logger=logger,
                mt_folder=join(folder, cli_deepl.model),
                num_of_sents=400,
                acceptable_range=(360, 480)
            )
            self.tasks[dm_id][cli_deepl.model] = task


if __name__ == '__main__':
    desc = '''Procedure 4 Task Manager
    Compute translations for 90 language pairs without English using DeepL
    Number of Input sentences: 400, within acceptable range of 360-480 sentences
    '''
    main(parser=proc_parser(desc=desc), proc=Proc6())
