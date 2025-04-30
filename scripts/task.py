from scripts.translators import translate_document, TranslationClient
from scripts.data_management import DataManager
from scripts.util import load_sents
from scripts.logger import MyLogger
import os
import time


class TranslationTask:
    def __init__(self, target_pairs: list[tuple[str, str]], dm: DataManager, client: TranslationClient, logger: MyLogger, mt_folder: str, num_of_sents: int, manual_retry: bool = False, max_retries=3, retry_delay=30):
        self.store = mt_folder
        self.pairs = [pair for pair in reversed(target_pairs)]
        self.dm = dm
        self.logger = logger
        self.num_of_sents = num_of_sents
        self.client = client
        os.makedirs(self.store, exist_ok=True)
        self.manual_retry = manual_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def run(self):
        retries = 0
        while len(self.pairs) > 0:
            pair = self.pairs.pop()
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

            if self.manual_retry:
                self.logger.add_retry_info(pair)

            try:
                translate_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    mt_folder=self.store,
                    client=self.client
                )
                retries = 0
            except Exception as e:
                self.logger.log_error(
                    error=e,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    translator=self.client.model
                )
                print('Error:\n', str(e))
                print(f'Waiting {self.retry_delay} seconds before retrying...')
                time.sleep(self.retry_delay)
                if retries < self.max_retries:
                    print('Retrying...')
                    retries += 1
                    self.pairs.append(pair)
                else:
                    print(
                        f'Failed {self.max_retries} times, skipping {pair}...')
                    retries = 0
                continue
            mt_sents = load_sents(self.store, src_lang, tgt_lang)
            print(f'{len(mt_sents)} translated from {src_lang} to {tgt_lang}')
