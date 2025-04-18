import tiktoken
import pandas as pd
from scripts.util import split_sents

# Cost: DEEPL_RATE * Char Count
DEEPL_RATE = 2e-5

# Cost for input and output tokens
GPT4o_RATE = (2.50 * 1e-6, 10.0 * 1e-6)

class TextStatistics:
    def __init__(self, sents : list[str], lang='unk', tokenizer='gpt-4o'):
        self.sents = sents
        real_sents = [split_sents(sent, lang) for sent in sents]
        self.num_of_sents = len(real_sents)
        self.enc = tiktoken.encoding_for_model(tokenizer)
        self.lang = lang
    
    def get_char_cnt(self):
        text = '\n'.join(self.sents)
        return len(text)
    
    def get_avg_char_cnt(self):
        return self.get_char_cnt() / self.num_of_sents
    
    def get_token_cnt(self):
        if not hasattr(self, '_token_cnt'):
            text = '\n'.join(self.sents)
            toks = self.enc.encode(text)
            self._token_cnt = len(toks)
        return self._token_cnt

    def get_avg_token_cnt(self):
        return self.get_token_cnt() / self.num_of_sents
    
    def deepl_cost(self):
        return DEEPL_RATE * self.get_char_cnt()
    
    def gpt4o_cost(self):
        in_tok_cost = self.get_token_cnt() * GPT4o_RATE[0]
        out_tok_cost = self.get_token_cnt() * 1.25 * GPT4o_RATE[1]
        return in_tok_cost + out_tok_cost
    
    def generate_stats(self, num_translations=10):
        data = dict()
        data['lang'] = self.lang
        data['char'] = self.get_char_cnt()
        data['tokens'] = self.get_token_cnt()
        data['sents'] = self.num_of_sents
        data['avg tok per sent'] = data['tokens'] / data['sents']
        data['deepl_cost'] = self.deepl_cost() * num_translations
        data['gpt4o_cost'] = self.gpt4o_cost() * num_translations
        df = pd.DataFrame([data])
        return df