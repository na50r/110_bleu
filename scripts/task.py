from scripts.translators import TranslationClient
from scripts.data_management import DataManager
from scripts.util import load_sents, split_sents, get_git_revision_short_hash, get_local_timestamp
from scripts.logger import TranslationLogger
import os
import time
from os.path import join, exists
from collections import defaultdict
import uuid
import logging
import json


class TranslationTask:
    '''
    Implementation of Translation Task for 110 BLEU project

    This class is used to run the translation task in batches. 
    Selection of language pairs, dataset, number of sentences and translator are the main configurable arguments.
    It is not possible to run tasks involving multiple datasets or multiple translators with this implementation.
    Error handling and rejection of insufficient or potentially malformed output is handled.
    In such cases, the API is called again automatically after specified delay for a specified number of times.
    '''

    def __init__(self, target_pairs: list[tuple[str, str]],
                 dm: DataManager,
                 client: TranslationClient,
                 logger: TranslationLogger,
                 mt_folder: str,
                 num_of_sents: int,
                 manual_retry: bool = False,
                 max_retries: int = 3,
                 retry_delay: int = 30,
                 acceptable_range: tuple[int, int] | None = None):
        '''
        Args:
            target_pairs: Selection of language pair to be translated
            dm: Selected DataManager, thus selected dataset, for this task
            client: Selected Translator for this task
            logger: A TranslationLogger that logs translation specific information
            mt_folder: Path to a folder where translations should be stored
            num_of_sents: Number of sentences to translate for each pair
            manual_retry: Set this to true if Task is run once again for selected pairs for which log_ids exist and reasons for retry exist
            max_retries: Maximum number of automatic retries for each translation, retries are triggered by errors or if the number of output sentences outside acceptable range
            retry_delay: Delay between retries in seconds
            acceptable_range: Range of acceptable number of output sentences, if None, it is set to 80% to 120% of num_of_sents
        '''

        self.id = str(uuid.uuid4())
        self.store = mt_folder
        self.pairs = [pair for pair in reversed(target_pairs)]
        self.dm = dm
        self.tl_logger = logger

        self.num_of_sents = num_of_sents
        self.client = client
        os.makedirs(self.store, exist_ok=True)
        self.acceptable_range = acceptable_range
        if self.acceptable_range is None:
            self.acceptable_range = (
                int(num_of_sents * 0.8), int(num_of_sents * 1.2))

        self.manual_retry = manual_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._retries = -1
        self._retry_pair = None
        self._fail_count = defaultdict(int)
        self._task_duration = None
        self._counter = 0

    def get_task_info(self):
        task_info = {}
        task_info['task_id'] = self.id
        task_info['git_hash'] = get_git_revision_short_hash()
        task_info['dataset'] = self.dm.name
        task_info['num_of_sents'] = self.num_of_sents
        task_info['split'] = self.dm.split
        task_info['translator'] = self.client.model
        task_info['acceptable_range'] = self.acceptable_range
        task_info['timestamp'] = get_local_timestamp()
        task_info['manual_retry'] = self.manual_retry
        return task_info

    def accept_output(self, mt_sents: list[str], tgt_lang: str):
        min_cnt = self.acceptable_range[0]
        max_cnt = self.acceptable_range[1]

        real_sents = split_sents(text='\n'.join(mt_sents), lang=tgt_lang)
        sent_cnt = len(real_sents)

        cond1 = len(mt_sents) >= min_cnt and len(mt_sents) <= max_cnt
        cond2 = sent_cnt >= min_cnt and sent_cnt <= max_cnt
        return cond1 or cond2

    def retry_loop(self, pair: tuple[str, str]):
        '''
        Decides when to retry and when to skip for a given pair
        '''
        if self._retry_pair != pair:
            self._retries = -1
            self._retry_pair = pair

        if self._retries == -1:
            self._retries += 1

        if self._retries < self.max_retries:
            logging.info(f'[⏲️]: Retrying {pair[0]}-{pair[1]}...')
            time.sleep(self.retry_delay)
            self._retries += 1
            self.pairs.append(pair)
            self.mark_failure(pair)
        else:
            logging.info(
                f'[⏩]: Failed {self.max_retries} times, skipping {pair[0]}-{pair[1]}...')
            self.mark_failure(pair)
            self._retries = -1

    def mark_failure(self, pair: tuple[str, str]):
        '''
        Renames files stored by translate_and_store_document for a given pair, to avoid file conflicts when retrying.
        '''
        self._fail_count[pair] += 1
        filename = join(self.store, f'{pair[0]}-{pair[1]}.txt')
        new_filename = join(
            self.store, f'{pair[0]}-{pair[1]}_fail{self._fail_count[pair]}.txt')
        if exists(filename):
            os.rename(filename, new_filename)

    def translation_id(self) -> str:
        return f'{self.id}-{self._counter:04d}'

    def finish(self):
        task_info = self.get_task_info()
        task_info['duration'] = self.duration
        with open(join(self.store, 'task.json'), 'w') as f:
            json.dump(task_info, f, indent=4)
        logging.info(
            f'[🏁]: Task took {self.duration:.2f}s')

    def run(self):
        start = time.time()
        logging.info(
            f'[🏁]: Starting task {self.id} on commit {get_git_revision_short_hash()}')
        while len(self.pairs) > 0:
            pair = self.pairs.pop()
            src_lang, tgt_lang = pair
            src_sents, _ = self.dm.get_sentence_pairs(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                num_of_sents=self.num_of_sents,
            )
            try:
                self._counter += 1
                mt_sents = self.client.translate_and_store_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    mt_folder=self.store,
                )

                self.tl_logger.add_entry(id=self.translation_id())
                self.tl_logger.add_entry(
                    translator=self.client.model, dataset=self.dm.name)
                if self.manual_retry:
                    self.tl_logger.add_manual_retry_info(pair)

                if self.accept_output(mt_sents, tgt_lang):
                    logging.info(
                        f'[✔️]: Translated {len(mt_sents)} sents for {src_lang}-{tgt_lang}')
                    mt_sents = load_sents(self.store, src_lang, tgt_lang)
                    self.tl_logger.write_log()
                    continue

                else:
                    logging.info(
                        f'[❌]: Translated {len(mt_sents)} sents for {src_lang}-{tgt_lang} but rejected')
                    self.tl_logger.write_log(verdict=False)
                    self.retry_loop(pair)
                    continue

            except Exception as e:
                logging.error(f'[⚠️]: Error {str(e)}')
                logging.debug("Traceback:", exc_info=True)
                self.retry_loop(pair)
                continue
        end = time.time()
        self.duration = end-start
        self.finish()
