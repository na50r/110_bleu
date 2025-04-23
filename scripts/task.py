from scripts.translators import translate_document
from scripts.data_management import FloresPlusManager, EPManager, Opus100Manager
from scripts.util import MyLogger, load_sents
import os

DM = {
    'ep': EPManager,
    'flores_plus': FloresPlusManager,
    'opus_100': Opus100Manager
}


class TranslationTask:
    def __init__(self, target_pairs: list[tuple[str, str]], dm: str, translator: str,  mt_folder: str, logfile: str, num_of_sents: int):
        self.store = mt_folder
        self.pairs = target_pairs
        self.dm = DM[dm]()
        self.dm_name = dm
        self.logger = MyLogger(logfile=logfile)
        self.num_of_sents = num_of_sents
        self.translator = translator
        os.makedirs(self.store, exist_ok=True)

    def run(self):
        for pair in self.pairs:
            src_lang, tgt_lang = pair
            src_sents, _ = self.dm.get_sentence_pairs(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                num_of_sents=self.num_of_sents,
            )
            self.logger.add_dataset_info(
                name=self.dm_name,
                num_of_sents=self.num_of_sents,
            )
            try:
                translate_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    logger=self.logger,
                    mt_folder=self.store,
                    translator=self.translator
                )
            except Exception as e:
                self.logger.log_error(
                    error=e,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    translator=self.translator
                )
                print(str(e))
                continue
            mt_sents = load_sents(self.store, src_lang, tgt_lang)
            print(f'{len(mt_sents)} translated from {src_lang} to {tgt_lang}')
