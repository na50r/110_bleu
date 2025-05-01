from scripts.translators import translate_document, TranslationClient
from scripts.data_management import DataManager
from scripts.util import load_sents, split_sents
from scripts.logger import MyLogger
import os
import time
from os.path import join, exists
from collections import defaultdict


class TranslationTask:
    def __init__(self, target_pairs: list[tuple[str, str]],
                 dm: DataManager,
                 client: TranslationClient,
                 logger: MyLogger,
                 mt_folder: str,
                 num_of_sents: int,
                 manual_retry: bool = False,
                 max_retries: int = 3,
                 retry_delay: int = 30):
        
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
        
        self.retries = -1
        self.retry_pair = None
        self.failure = defaultdict(int)
        
    def accept_output(self, mt_sents : list[str], tgt_lang: str):
        min_cnt = self.num_of_sents * 0.9
        max_cnt = self.num_of_sents * 1.1
        
        real_sents = split_sents(text='\n'.join(mt_sents), lang=tgt_lang)
        sent_cnt = len(real_sents)
        
        cond1 = len(mt_sents) >= min_cnt and len(mt_sents) <= max_cnt
        cond2 = sent_cnt >= min_cnt and sent_cnt <= max_cnt       
        return cond1 or cond2
    
    def retry_loop(self, pair):
        if self.retry_pair != pair:
            self.retries = -1
            self.retry_pair = pair
        
        if self.retries == -1:
            self.retries+=1
        
        if self.retries < self.max_retries:
            print(f'Retrying {pair}...')
            time.sleep(self.retry_delay)
            self.retries += 1
            self.pairs.append(pair)
            self.mark_failure(pair)
        else:
            print(f'Failed {self.max_retries} times, skipping {pair}...')
            self.mark_failure(pair)
            self.retries = -1

    def mark_failure(self, pair):
        self.failure[pair] += 1
        filename = join(self.store, f'{pair[0]}-{pair[1]}.txt')
        new_filename = join(self.store, f'{pair[0]}-{pair[1]}_fail{self.failure[pair]}.txt')
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
                self.logger.add_retry_info(pair)

            try:
                mt_sents = translate_document(
                    text=src_sents,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    mt_folder=self.store,
                    client=self.client
                )
                
                if self.accept_output(mt_sents, tgt_lang):
                    print(f'{len(mt_sents)} translated from {src_lang} to {tgt_lang}')
                    mt_sents = load_sents(self.store, src_lang, tgt_lang)
                    self.logger.add_entry(verdict={'success': 'Translation accepted'})
                    self.logger.write_log()
                    continue
                
                else:
                    print(f'Output for {pair} is not acceptable!')
                    self.logger.add_entry(verdict={'failure': 'Translation rejected'})
                    self.logger.write_log()
                    self.retry_loop(pair)
                    continue 
                
            except Exception as e:
                self.logger.log_error(error=e)
                print('Error:\n', str(e))
                self.retry_loop(pair)
                continue