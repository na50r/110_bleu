# NOTE: This code is still work in progress!
import numpy as np
import json
import pandas as pd
import random
import pickle
from sacrebleu.compat import corpus_bleu, corpus_chrf


def compute_bleu(ref: list[str], hyp: list[str]):
    refs = [ref]
    res = corpus_bleu(hyp, refs)
    return res.score


def compute_chrf(ref: list[str], hyp: list[str]):
    refs = [ref]
    res = corpus_chrf(hyp, refs)
    return res.score


def compute_comet_kiwi(data, lang):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    data = [{'src': o['src'], 'mt': o['mt']} for o in data]
    model_output = model.predict(data, batch_size=8, gpus=1)
    print(f'Computed COMET Kiwi scores for src & mt for lang {lang}')
    return model_output


def comet_kiwi_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        tup = (o['mt'], o['src'])
        mapping[tup] = s
    return mapping


def pre_compute_comet_kiwi(data, mapping):
    scores = []
    for o in data:
        tup = (o['mt'], o['src'])
        s = mapping[tup]
        scores.append(s)
    return np.array(scores).mean() * 100


def compute_comet(data, lang):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data, batch_size=8, gpus=1)
    print(f'Computed COMET scores for ref & hyp of lang {lang}')
    return model_output


def comet_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        triple = (o['ref'], o['mt'], o['src'])
        mapping[triple] = s
    return mapping


def pre_compute_comet(data, mapping):
    scores = []
    for o in data:
        triple = (o['ref'], o['mt'], o['src'])
        s = mapping[triple]
        scores.append(s)
    return np.array(scores).mean() * 100


def compute_bert_score(ref, hyp, lang):
    # Recale with Baseline set to True as specified in: https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
    from bert_score import score
    P, R, F1 = score(hyp, ref, lang=lang, verbose=False,
                     rescale_with_baseline=True)
    print(f'Computed BERT-F1 scores for ref & hyp of lang {lang}')
    return F1


def bert_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        tup = (o['ref'], o['mt'])
        mapping[tup] = s
    return mapping


def pre_compute_bert(data, mapping):
    # Pre-computes BERT-F1 scores after computed once
    # Can be used to compute BERT-F1 scores of smaller samples
    scores = []
    for o in data:
        tup = (o['ref'], o['mt'])
        s = mapping[tup]
        scores.append(s)
    return np.array(scores).mean() * 100


class ResultProducer:
    def __init__(self, BLEU_func=compute_bleu, chrF_func=compute_chrf, label2files=None, use_bert=False, use_comet=False, use_comet_kiwi=False):
        self.BLEU = BLEU_func
        self.chrF = chrF_func
        self.BERT = use_bert
        self.COMET = use_comet
        self.COMET_KIWI = use_comet_kiwi
        self.label2files = label2files
        self.data_set_sizes = {}
        self.comet_mapping = {}
        self.bert_mapping = {}
        self.bleu_scores = []
        self.bert_scores = []
        self.comet_scores = []
        self.chrf_scores = []

        self.comet_kiwi_scores = []
        self.comet_kiwi_mapping = {}

    def clear_mappings(self):
        self.comet_mapping = {}
        self.bert_mapping = {}

    def compute_results(self, randomize=False, split_value=None):
        '''
        Computes the BLEU, chrF, BERT-F1 and Comet scores of a given JSONL file/s of aligned sentences
        Expected format: {src : src_sent, ref : ref_sent, mt : mt_sent} (Based on Comet API specification)

        Args:
            randomize: Randomizes order of provided sentences
            split_value: Must be a value below 1, shrinks the number of sentences accoringly, used for sampling
            swap_mt_ref : Swaps the mt and ref sentences 
        '''

        # Clear scores each time this method is called!
        self.bert_scores = []
        self.bleu_scores = []
        self.comet_scores = []
        self.chrf_scores = []
        self.comet_kiwi_scores = []

        for lang in self.label2files:
            file = self.label2files[lang]
            with open(file, 'r') as f:
                data = [json.loads(ln) for ln in f]

            if randomize:
                random.shuffle(data)

            if split_value != None:
                split_point = int(len(data) * split_value)
                data = data[:split_point]

            self.data_set_sizes[lang] = len(data)
            # Store meta data
            mt_sents = [o['mt'] for o in data]
            ref_sents = [o['ref'] for o in data]
            # Eager evaluate BLEU and chrF scores
            self.bleu_scores.append(self.BLEU(ref_sents, mt_sents))
            self.chrf_scores.append(self.chrF(ref_sents, mt_sents))

            # BERT score optional
            # bert_mapping used for recomputation of sampled or randomized data
            # Avoids encoding the sentences again and again
            if self.BERT == True:
                if lang not in self.bert_mapping:
                    bert_out = compute_bert_score(ref_sents, mt_sents, lang)
                    mapping = bert_mapper(data, bert_out)
                    self.bert_mapping[lang] = mapping

                    # BERT-F1 score computed as value between 0 to 1
                    # To make Table cleaner, we adhere to BLEU & chrF score and make it value between 0 and 100
                    self.bert_scores.append(np.array(bert_out).mean() * 100)
                else:
                    self.bert_scores.append(pre_compute_bert(
                        data, self.bert_mapping[lang]))

            if self.COMET == True:
                if lang not in self.comet_mapping:
                    comet_out = compute_comet(data, lang)
                    mapping = comet_mapper(data, comet_out['scores'])
                    self.comet_mapping[lang] = mapping
                    self.comet_scores.append(comet_out['system_score']*100)
                else:
                    self.comet_scores.append(pre_compute_comet(
                        data, self.comet_mapping[lang]))

            if self.COMET_KIWI == True:
                if lang not in self.comet_kiwi_mapping:
                    comet_out = compute_comet_kiwi(data, lang)
                    mapping = comet_kiwi_mapper(data, comet_out['scores'])
                    self.comet_kiwi_mapping[lang] = mapping
                    self.comet_kiwi_scores.append(
                        comet_out['system_score']*100)
                else:
                    self.comet_kiwi_scores.append(pre_compute_comet_kiwi(
                        data, self.comet_kiwi_mapping[lang]
                    ))

    def _get_scores(self):
        SCORES = {
            'Label': self.label2files.keys(),
            'BLEU': self.bleu_scores,
            'chrF': self.chrf_scores,
        }
        if len(self.bert_scores) != 0:
            SCORES.update({'BERT-F1': self.bert_scores})

        if len(self.comet_scores) != 0:
            SCORES.update({'COMET': self.comet_scores})

        if len(self.comet_kiwi_scores) != 0:
            SCORES.update({'COMET_KIWI': self.comet_kiwi_scores})

        return SCORES

    def display_results(self, latex=False):
        SCORES = self._get_scores()
        if len(self.bert_scores) != 0:
            SCORES.update({'BERT-F1': self.bert_scores})

        if len(self.comet_scores) != 0:
            SCORES.update({'COMET': self.comet_scores})

        if len(self.comet_kiwi_scores) != 0:
            SCORES.update({'COMET_KIWI': self.comet_kiwi_scores})
        df = pd.DataFrame(SCORES)

        if latex:
            latex_table = df.to_latex(
                index=False, float_format="%.2f")
            print(latex_table)
        print(df)

    def store_results(self, output_path):
        SCORES = self._get_scores()
        df = pd.DataFrame(SCORES)
        df.to_csv(output_path, index=False)

    def store_mappings(self, output_path: str):
        """
        Store both BERT and Comet mappings from pickle file

        Args:
            output_path (str): Full path to the output pickle file
        """
        # Create mapping dictionary to store
        mappings = {
            'bert_mapping': self.bert_mapping,
            'comet_mapping': self.comet_mapping,
            'comet_kiwi_mapping': self.comet_kiwi_mapping
        }
        # Store mappings
        with open(output_path, 'wb') as f:
            pickle.dump(mappings, f)

        print(f"Mappings stored successfully in {output_path}")

    def load_mappings(self, input_path: str):
        """
        Load both BERT and Comet mappings from a pickle file.

        Args:
            input_path: Full path to the input pickle file
        """
        # Load mappings
        with open(input_path, 'rb') as f:
            mappings = pickle.load(f)

        # Restore mappings
        self.bert_mapping = mappings.get('bert_mapping', {})
        self.comet_mapping = mappings.get('comet_mapping', {})
        self.comet_kiwi_mapping = mappings.get('comet_kiwi_mapping', {})

        print(f"Mappings loaded successfully from {input_path}")
        print(
            f"Loaded BERT mappings for languages: {list(self.bert_mapping.keys())}")
        print(
            f"Loaded COMET mappings for languages: {list(self.comet_mapping.keys())}")
        print(
            f"Loaded COMET KIWI mappings for languages: {list(self.comet_kiwi_mapper.keys())}"
        )


def create_matrix_from_csv(path_to_csv, metric='BLEU'):
    # https://stackoverflow.com/questions/72827153/how-to-extract-specific-key-and-value-from-a-dataframe-python
    '''
    Assume CSV of format:
    Label   Metric
    src-tgt value

    I.e.
    Label   BLEU
    de-en   31.04
    en-de   28.89
    ...
    '''
    df = pd.read_csv(path_to_csv)
    all_langs = []
    for l in df.Label:
        src, tgt = l.split('-')
        all_langs.append(src)
        all_langs.append(tgt)
    all_langs = sorted(list(set(all_langs)))

    df.set_index('Label', inplace=True)
    label2metric = df[metric].to_dict()

    lang2metric = {l: [] for l in all_langs}
    for src in all_langs:
        for tgt in all_langs:
            pair = f'{src}-{tgt}'
            value = label2metric.get(pair, None)
            lang2metric[src].append(value)

    df_tgt_src = pd.DataFrame(data=lang2metric, index=all_langs)
    df_src_tgt = df_tgt_src.T
    return df_src_tgt
