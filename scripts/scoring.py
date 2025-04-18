import numpy as np
import json
import pandas as pd
import random
import pickle
import os
from os.path import join
import re
from sacrebleu.compat import corpus_bleu, corpus_chrf

# BLEU & chrF computation


def compute_bleu(ref: list[str], hyp: list[str]):
    refs = [ref]
    res = corpus_bleu(hyp, refs)
    return res.score


def compute_chrf(ref: list[str], hyp: list[str]):
    refs = [ref]
    res = corpus_chrf(hyp, refs)
    return res.score


# CometKiwi
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

# Comet computation


def compute_comet(data, lang):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    model_output = model.predict(data, batch_size=8, gpus=1)
    print(f'Computed COMET scores for ref & hyp of lang {lang}')
    return model_output

# Store comet results after computed once


def comet_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        triple = (o['ref'], o['mt'], o['src'])
        mapping[triple] = s
    return mapping

# Precomputes Comet scores based on stored results
# Can be used to compute Comet scores of smaller samples


def pre_compute_comet(data, mapping):
    scores = []
    for o in data:
        triple = (o['ref'], o['mt'], o['src'])
        s = mapping[triple]
        scores.append(s)
    return np.array(scores).mean() * 100


# BERT scores
# Recale with Baseline set to True as specified in: https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
def compute_bert_score(ref, hyp, lang):
    from bert_score import score
    P, R, F1 = score(hyp, ref, lang=lang, verbose=False,
                     rescale_with_baseline=True)
    print(f'Computed BERT-F1 scores for ref & hyp of lang {lang}')
    return F1

# Stores BERT-F scores after computed once


def bert_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        tup = (o['ref'], o['mt'])
        mapping[tup] = s
    return mapping

# Pre-computes BERT-F1 scores after computed once
# Can be used to compute BERT-F1 scores of smaller samples


def pre_compute_bert(data, mapping):
    scores = []
    for o in data:
        tup = (o['ref'], o['mt'])
        s = mapping[tup]
        scores.append(s)
    return np.array(scores).mean() * 100

# Overall class to produce results


class ResultProducer:
    def __init__(self, BLEU_func=compute_bleu, chrF_func=compute_chrf, lang2files=None, use_bert=False, use_comet=False, use_comet_kiwi=False):
        self.BLEU = BLEU_func
        self.chrF = chrF_func
        self.BERT = use_bert
        self.COMET = use_comet
        self.COMET_KIWI = use_comet_kiwi
        self.lang2files = lang2files
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

        for lang in self.lang2files:
            file = self.lang2files[lang]
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
            'Src': self.lang2files.keys(),
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

    def display_results(self, tgt='Unknown', latex=False):
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

# Code produced with the help of ChatGPT4o


def create_matrix_from_csv(csv_folder, metric):
    csv_files = [join(csv_folder, f) for f in os.listdir(csv_folder)]

    # Store all rows here
    records = []
    for file in csv_files:
        # Get target lang from filename, e.g. de2en.csv -> 'en'
        tgt = re.search(r'(\w\w).csv$', file).group(1)
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            records.append({
                'Src': row['Src'],
                'Tgt': tgt,
                metric: row[metric]
            })

    all_scores = pd.DataFrame(records)

    # Pivot into matrix: rows=Src, columns=Tgt, values=metric
    score_matrix = all_scores.pivot(
        index='Src', columns='Tgt', values=metric).round(2)

    # Optionally sort rows and columns
    score_matrix = score_matrix.sort_index().sort_index(axis=1)
    return score_matrix


def score_matrix_to_latex(matrix):
    latext_table = matrix.round(2).to_latex(
        index=True,
        na_rep='--',           # How to display missing values
        # left-align row names, center columns
        column_format='l' + 'c' * len(matrix.columns),
        bold_rows=True,
        escape=False,
        multicolumn=True,
        multicolumn_format='c',
        float_format="%.2f"
    )
    print(latext_table)
