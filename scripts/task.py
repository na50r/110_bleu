from scripts.translators import translate_document, TranslationClient
from scripts.data_management import DataManager
from scripts.util import load_sents
from scripts.logger import MyLogger
import os


class TranslationTask:
    def __init__(self, target_pairs: list[tuple[str, str]], dm: DataManager, client: TranslationClient, logger: MyLogger, mt_folder: str, num_of_sents: int, is_retry: bool = False):
        self.store = mt_folder
        self.pairs = target_pairs
        self.dm = dm
        self.logger = logger
        self.num_of_sents = num_of_sents
        self.client = client
        os.makedirs(self.store, exist_ok=True)
        self.is_retry = is_retry

    def run(self):
        for pair in self.pairs:
            src_lang, tgt_lang = pair
            src_sents, _ = self.dm.get_sentence_pairs(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                num_of_sents=self.num_of_sents,
            )
            self.logger.add_dataset_info(
                name=self.dm.name,
                num_of_sents=self.num_of_sents,
                split=self.dm.split,
            )

            if self.is_retry:
                self.logger.add_retry_info(pair)

            try:
                translate_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    mt_folder=self.store,
                    client=self.client
                )
            except Exception as e:
                self.logger.log_error(
                    error=e,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    translator=self.client.model
                )
                print(str(e))
                continue
            mt_sents = load_sents(self.store, src_lang, tgt_lang)
            print(f'{len(mt_sents)} translated from {src_lang} to {tgt_lang}')
