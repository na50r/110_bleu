from scripts.translators import translate_document, TranslationClient
from scripts.data_management import DataManager
from scripts.util import load_sents, split_sents
from scripts.logger import MyLogger
import os
import time
from os.path import join, exists
from collections import defaultdict


class TranslationTask:
    '''
    Implementation of Translation Task for 110 BLEU project

    This class is used to run the translation task in batches. 
    Selection of language pairs, dataset, number of sentences and translator are the main configurable arguments.
    It is not possible to run tasks involving multiple datasets or multiple translators with this implementation.
    Error handling and rejection of insufficient or potentially malformed output is also handled.
    In such cases, the API is called again automatically after specified delay for a specified number of times.

    Attributes:
        store: Path to a folder where translations should be stored
        pairs: A list of tuples of source and target language ISO codes (acts as a stack but preserves insertion order)
        dm: A data manager that has a get_sentence_pairs method specified by DataManager abstract class
        client: A client that has a translate_document method specified by TranslationClient abstract class
        logger: A logger that is used to log task/input related configuration and each translation
        num_of_sents: Number of sentences to translate for each pair
        manual_retry: Boolean to indicate whether this is a manual retry task or not
        max_retries: Maximum number of automatic retries for each translation
        retry_delay: Delay between retries in seconds
        acceptable_range: Tuple containing min and max integer values for acceptable number of output sentences
        retries: Number of retries for current pair
        retry_pair: Current pair for which retries are being performed
        failure: Dictionary that keeps track of number of failures for each pair
    '''

    def __init__(self, target_pairs: list[tuple[str, str]],
                 dm: DataManager,
                 client: TranslationClient,
                 logger: MyLogger,
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
            logger: Logger to log task/input related configuration and each translation, should be the passed into TranslationClient as well
            mt_folder: Path to a folder where translations should be stored
            num_of_sents: Number of sentences to translate for each pair
            manual_retry: Set this to true if Task is run once again for selected pairs for which log_ids exist and reasons for retry exist
            max_retries: Maximum number of automatic retries for each translation, retries are triggered by errors or if the number of output sentences outside acceptable range
            retry_delay: Delay between retries in seconds
            acceptable_range: Range of acceptable number of output sentences
        '''

        self.store = mt_folder
        self.pairs = [pair for pair in reversed(target_pairs)]
        self.dm = dm
        self.logger = logger
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

        self.retries = -1
        self.retry_pair = None
        self.failure = defaultdict(int)

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
        if self.retry_pair != pair:
            self.retries = -1
            self.retry_pair = pair

        if self.retries == -1:
            self.retries += 1

        if self.retries < self.max_retries:
            print(f'[⏲️]: Retrying {pair[0]}-{pair[1]}...')
            time.sleep(self.retry_delay)
            self.retries += 1
            self.pairs.append(pair)
            self.mark_failure(pair)
        else:
            print(
                f'[⏩]: Failed {self.max_retries} times, skipping {pair[0]}-{pair[1]}...')
            self.mark_failure(pair)
            self.retries = -1

    def mark_failure(self, pair: tuple[str, str]):
        '''
        Renames files stored by translate_document for a given pair. to avoid file conflicts when retrying.
        '''
        self.failure[pair] += 1
        filename = join(self.store, f'{pair[0]}-{pair[1]}.txt')
        new_filename = join(
            self.store, f'{pair[0]}-{pair[1]}_fail{self.failure[pair]}.txt')
        if exists(filename):
            os.rename(filename, new_filename)

    def run(self):
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
                self.logger.add_manual_retry_info(pair)

            try:
                mt_sents = translate_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    mt_folder=self.store,
                    client=self.client
                )

                if self.accept_output(mt_sents, tgt_lang):
                    print(
                        f'[✔️]: {len(mt_sents)} translated from {src_lang} to {tgt_lang}')
                    mt_sents = load_sents(self.store, src_lang, tgt_lang)
                    self.logger.add_entry(
                        verdict={'success': 'Translation accepted'})
                    self.logger.write_log()
                    continue

                else:
                    print(
                        f'[❌]: Output for {src_lang}-{tgt_lang} is not acceptable!')
                    self.logger.add_entry(
                        verdict={'failure': 'Translation rejected'})
                    self.logger.write_log()
                    self.retry_loop(pair)
                    continue

            except Exception as e:
                self.logger.log_error(error=e)
                print('[⚠️]: Error', str(e))
                self.retry_loop(pair)
                continue
