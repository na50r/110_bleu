from scripts.data_management import FloresPlusManager, Opus100Manager, EPManager
import random
# Run tests with: python -m pytest --capture=tee-sys tests.py

def test_flores_1():
    dm = FloresPlusManager()
    expected = random.randint(0, 100)
    lang = random.sample(sorted(FloresPlusManager.EURO_LANGS), k=1)[0]
    out = dm.get_sentences(lang, num_of_sents=expected)
    assert len(out[lang]) == expected

def test_flores_2():
    dm = FloresPlusManager()
    expected = random.randint(0, 100)
    src_lang, tgt_lang = random.sample(sorted(FloresPlusManager.EURO_LANGS), 2)
    src_sents, tgt_sents = dm.get_sentence_pairs(src_lang, tgt_lang, num_of_sents=expected)
    assert len(src_sents) == expected
    assert len(tgt_sents) == expected

def test_opus_1():
    dm = Opus100Manager()
    expected = random.randint(0, 100)
    lang = random.sample(sorted(Opus100Manager.EURO_LANGS), k=1)[0]
    src_sents, tgt_sents = dm.get_sentence_pairs(lang, 'en', num_of_sents=expected)
    assert len(src_sents) == expected
    assert len(tgt_sents) == expected
    
    
def test_ep_1():
    dm = EPManager()
    expected = random.randint(0, 100)
    src_lang, tgt_lang = random.sample(sorted(EPManager.EURO_LANGS), 2)
    src_sents, tgt_sents = dm.get_sentence_pairs(
        src_lang, tgt_lang, num_of_sents=expected)
    assert len(src_sents) == expected
    assert len(tgt_sents) == expected
