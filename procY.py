from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config
from scripts.translators import MockClient
logging_config('procY.log')


class ProcY(Procedure):
    def __init__(self):
        # Define all English including pairs
        en_pairs = Opus100Manager.get_pairs()
        all_pairs = EuroParlManager.get_pairs()
        non_en_pairs = set(all_pairs) - set(en_pairs)
        assert len(non_en_pairs) == 90

        # Define folder hierarchy of where translations should be stored
        main_folder = 'tmp_tasks'
        sub_folder = join(main_folder, 'procY')

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager()]
        self.dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in self.dm_ids]
        self.tasks = {dm_id: {} for dm_id in self.dm_ids}

        # Define logger and clients
        logger = TranslationLogger(logfile=join(main_folder, 'procY.jsonl'))
        cli_deepl = MockClient(logger=logger, model='deepl_document')
        self.model_ids = [cli_deepl.model]

        for dm, folder, dm_id in zip(dms, dm_folders, self.dm_ids):
            task = TranslationTask(
                target_pairs=list(non_en_pairs),
                dm=dm,
                client=cli_deepl,
                logger=logger,
                mt_folder=join(folder, cli_deepl.model),
                num_of_sents=400,
                acceptable_range=(360, 480),
                lang_detection=False,
            )
            self.tasks[dm_id][cli_deepl.model] = task


if __name__ == '__main__':
    desc = '''Procedure 2 Task Manager
    Compute translations for 90 language pairs without English using DeepL
    Number of Input sentences: 400, within acceptable range of 360-480 sentences
    '''
    main(parser=proc_parser(desc=desc), proc=ProcY())
