import tiktoken
from scripts.util import split_sents

# Cost: DEEPL_RATE * Char Count
DEEPL_RATE = 2e-5

# Cost for input and output tokens
GPT4o_RATE = (2.50 * 1e-6, 10.0 * 1e-6)

# Cost for input and output tokens
GPT41_RATE = (2.00 * 1e-6, 8.0 * 1e-6)

ENC = tiktoken.encoding_for_model('gpt-4o')

def get_char_cnt(input_sents : list[str]):
    text = '\n'.join(input_sents)
    return len(text)

def get_deepl_cost(input_sents : list[str]):
    '''
    Cost for translating input_sents into a different, single language
    '''
    return get_char_cnt(input_sents) * DEEPL_RATE

def get_token_cnt(input_sents : list[str]):
    text = '\n'.join(input_sents)
    return len(ENC.encode(text))

def get_gpt41_cost(input_sents : list[str]):
    '''
    Cost for translation input_sents into a different, single language
    '''
    tok_cnt = get_token_cnt(input_sents)
    return GPT41_RATE[0] * tok_cnt + GPT41_RATE[1] * tok_cnt

def get_real_sent_cnt(input_sents : list[str], lang):
    text = '\n'.join(input_sents)
    return len(split_sents(text, lang=lang))