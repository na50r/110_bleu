from scripts.procedure import main, proc_parser, Procedure
from os.path import join
from scripts.data_management import EuroParlManager, Opus100Manager, FloresPlusManager
from scripts.task import TranslationTask
from scripts.logger import TranslationLogger, logging_config
from scripts.translators import GPTClient
logging_config('proc3.log')


class Proc3(Procedure):
    def __init__(self):
        en_pairs = Opus100Manager.get_pairs()
        all_pairs = EuroParlManager.get_pairs()
        non_en_pairs = set(all_pairs) - set(en_pairs)
        non_en_pairs = list(sorted(non_en_pairs))
        langs = sorted(Opus100Manager.EURO_ISO_2_PAIR.keys())
        assert len(non_en_pairs) == 90

        segmented = {}
        for lang in langs:
            segmented[lang] = [p for p in non_en_pairs if p[0] == lang]
        assert len(segmented) == 10

        # Define folder hierarchy of where translations should be stored
        main_folder = 'tasks'
        sub_folder = join(main_folder, 'proc3')
        self.tasks = {}

        # Define the data managers and folders for translation storage
        dms = [EuroParlManager(), FloresPlusManager()]
        dm_ids = [dm.short_name for dm in dms]
        dm_folders = [join(sub_folder, dm_id) for dm_id in dm_ids]
        for dm_id in dm_ids:
            for s in segmented:
                self.tasks[f'{dm_id}-{s}'] = {}

        # Tasks that have been completed already by proc2.py
        completed = [
            'europarl-da',
            'europarl-de',
            'europarl-el',
            'europarl-es',
            'europarl-fi'
        ]
        for c in completed:
            del self.tasks[c]

        # Define logger and clients
        logger = TranslationLogger(logfile=join(main_folder, 'proc3.jsonl'))
        cli = GPTClient(logger=logger)
        self.model_ids = [cli.model]
        self.dm_ids = list(self.tasks.keys())

        for dm, folder, dm_id in zip(dms, dm_folders, dm_ids):
            for s in segmented:
                key = f'{dm_id}-{s}'
                if key in completed:
                    continue
                task = TranslationTask(
                    target_pairs=segmented[s],
                    dm=dm,
                    client=cli,
                    logger=logger,
                    mt_folder=join(f'{folder}-{s}', cli.model),
                    num_of_sents=400,
                    acceptable_range=(360, 480),
                    lang_detection=False,
                )
                self.tasks[key][cli.model] = task


if __name__ == '__main__':
    desc = '''Procedure 3 Task Manager
    Compute translations for 90 language pairs without English using GPT4.1
    Translations in batches of 9 (src language), splitting 90 pairs from EuroParl and Flores+ dataset into 10 batches each
    We decided to implement this after running the first half of EuroParl on proc2.py and realizing that it takes too much time per run.
    There was also a bug in proc2.py which was taken care of here. 
    Number of Input sentences: 400, within acceptable range of 360-480 sentences
    '''
    main(parser=proc_parser(desc=desc), proc=Proc3())
