from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config
from scripts.translators import GPTClient
logging_config('proc2.log')


class Proc2(Procedure):
    def __init__(self):
        en_pairs = Opus100Manager.get_pairs()
        all_pairs = EuroParlManager.get_pairs()
        non_en_pairs = set(all_pairs) - set(en_pairs)
        non_en_pairs = list(sorted(non_en_pairs))
        langs = sorted(Opus100Manager.EURO_ISO_2_PAIR.keys())
        assert len(non_en_pairs) == 90

        segmented = []
        for lang in langs:
            segmented.append([p for p in non_en_pairs if p[0]==lang])
        
        assert len(segmented)==10
        half1 = []
        half2 = []
        for s in segmented[:5]:
            half1.extend(s)
        for s in segmented[5:]:
            half2.extend(s)
        
        assert len(half1) == 45
        assert len(half2) == 45
        halves = [half1, half2]

        # Define folder hierarchy of where translations should be stored
        main_folder = 'tasks'
        sub_folder = join(main_folder, 'proc2')
        self.tasks = {}

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager()]
        dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in dm_ids]
        for dm_id in dm_ids:
            for i in range(1, 3):
                self.tasks[f'{dm_id}-{i}'] = {}

        self.dm_ids = list(self.tasks.keys())
        # Define logger and clients
        logger = TranslationLogger(logfile=join(main_folder, 'proc2.jsonl'))
        cli = GPTClient(logger=logger)
        self.model_ids = [cli.model]

        for dm, folder, dm_id in zip(dms, dm_folders, dm_ids):
            for i, half in enumerate(halves):
                task = TranslationTask(
                    target_pairs=list(half),
                    dm=dm,
                    client=cli,
                    logger=logger,
                    mt_folder=join(folder, cli.model),
                    num_of_sents=400,
                    acceptable_range=(360, 480)
                )
                self.tasks[f'{dm_id}-{i+1}'][cli.model] = task


if __name__ == '__main__':
    desc = '''Procedure 2 Task Manager
    Compute translations for 90 language pairs without English using GPT4.1
    Translations in batches of 45, splitting 90 pairs from EuroParl and Flores+ dataset into 2 batches each
    Number of Input sentences: 400, within acceptable range of 360-480 sentences
    '''
    main(parser=proc_parser(desc=desc), proc=Proc2())
