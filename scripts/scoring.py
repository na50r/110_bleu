# NOTE: This code is still work in progress!
import numpy as np
import json
import pandas as pd
import random
import pickle
from sacrebleu.compat import corpus_bleu, corpus_chrf
import logging
import time


def compute_bleu(ref: list[str], hyp: list[str]):
    refs = [ref]  # Because only single reference
    res = corpus_bleu(hyp, refs)
    return res.score


def compute_chrf(ref: list[str], hyp: list[str]):
    refs = [ref]  # Because only single reference
    res = corpus_chrf(hyp, refs)
    return res.score


def load_comet_model():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    return model


def compute_comet(data: dict[str, str], model):
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output


def comet_mapper(data: dict[str, str], scores: list[float]):
    mapping = {}
    for o, s in zip(data, scores):
        triple = (o['ref'], o['mt'], o['src'])
        mapping[triple] = s
    return mapping


def precompute_comet(data: dict[str, str], mapping: dict[tuple[str, str, str], float]):
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
    return F1


def bert_mapper(data, scores):
    mapping = {}
    for o, s in zip(data, scores):
        tup = (o['ref'], o['mt'])
        mapping[tup] = s
    return mapping


def precompute_bert(data, mapping):
    # Precomputes BERT-F1 scores after computed once
    # Can be used to compute BERT-F1 scores of smaller samples
    scores = []
    for o in data:
        tup = (o['ref'], o['mt'])
        s = mapping[tup]
        scores.append(s)
    return np.array(scores).mean() * 100


class ResultProducer:
    def __init__(self, label2files: dict[str, str] = None, use_bert: bool = False, use_comet: bool = False, save_mappings: bool = False):
        '''
        Args:
            label2files: A dictionary that uses labels as keys and filepath to a JSONL file of COMET format {"mt":"sent", "ref":"sent", "src":"sent"}
            use_bert: A boolean, decide if you want to compute BERT-F1 score or not, not recommended for CPU
            use_comet: A boolean, decide if you want to compute COMET score or not, not recommended for CPU
        '''
        self.use_bert = use_bert
        self.use_comet = use_comet
        self.label2files = label2files
        self.data_set_sizes = {}
        self.comet_mapping = {}
        self.bert_mapping = {}
        self.bleu_scores = []
        self.bert_scores = []
        self.comet_scores = []
        self.chrf_scores = []
        if self.use_comet:
            self.comet_model = load_comet_model()
        self.save_mappings = save_mappings

    def clear_mappings(self):
        self.comet_mapping = {}
        self.bert_mapping = {}

    def compute_results(self, randomize: bool = False, split_point: int = None):
        '''
        Computes the BLEU, chrF, BERT-F1 and Comet scores of a given JSONL file/s of aligned sentences
        Expected format: {src : src_sent, ref : ref_sent, mt : mt_sent} (Based on Comet API specification)
        Label contains suffix of mt/ref language! (For BERT-F1 Score), i.e. ep-gpt-de-en.jsonl or de-en.jsonl must contain English translations & references

        Args:
            randomize: Randomizes order of provided sentences
            split_value: Must be a value below 1, shrinks the number of sentences accoringly, used for sampling
        '''

        # Clear scores each time this method is called!
        self.bert_scores = []
        self.bleu_scores = []
        self.comet_scores = []
        self.chrf_scores = []

        for label in self.label2files:
            file = self.label2files[label]
            with open(file, 'r') as f:
                data = [json.loads(ln) for ln in f]

            if randomize:
                random.shuffle(data)

            if split_point is not None:
                assert split_point <= len(data), 'Choose smaller split point'
                data = data[:split_point]

            self.data_set_sizes[label] = len(data)
            # Store meta data
            mt_sents = [o['mt'] for o in data]
            ref_sents = [o['ref'] for o in data]
            logging.info(
                f'Start scoring for {label} with {len(data)} triplets')
            # Eager evaluate BLEU and chrF scores
            self.bleu_scores.append(compute_bleu(ref_sents, mt_sents))
            self.chrf_scores.append(compute_chrf(ref_sents, mt_sents))

            # BERT score optional
            # bert_mapping used for recomputation of sampled or randomized data
            # Avoids encoding the sentences again and again
            if self.use_bert == True:
                if label not in self.bert_mapping:
                    # Assume label de-en, so [-1] == 'en' == target language
                    lang = label.split('-')[-1]
                    start = time.time()
                    bert_out = compute_bert_score(ref_sents, mt_sents, lang)
                    end = time.time()
                    logging.info(
                        f'BERTScore took {end-start:.2f} seconds for {label}')
                    if self.save_mappings:
                        mapping = bert_mapper(data, bert_out)
                        self.bert_mapping[label] = mapping

                    # BERT-F1 score computed as value between 0 to 1
                    # To make results cleaner, we adhere to BLEU & chrF and make it value between 0 and 100
                    self.bert_scores.append(np.array(bert_out).mean() * 100)
                else:
                    self.bert_scores.append(precompute_bert(
                        data, self.bert_mapping[label]))

            if self.use_comet == True:
                if label not in self.comet_mapping:
                    start = time.time()
                    comet_out = compute_comet(data, self.comet_model)
                    end = time.time()
                    logging.info(
                        f'COMET took {end-start:.2f} seconds for {label}')
                    if self.save_mappings:
                        mapping = comet_mapper(data, comet_out['scores'])
                        self.comet_mapping[label] = mapping
                    self.comet_scores.append(comet_out['system_score']*100)
                else:
                    self.comet_scores.append(precompute_comet(
                        data, self.comet_mapping[label]))

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

        return SCORES

    def get_scores(self):
        SCORES = self._get_scores()
        df = pd.DataFrame(SCORES)
        return df

    def display_results(self, latex: bool = False):
        df = self.get_scores()
        if latex:
            latex_table = df.to_latex(
                index=False, float_format="%.2f")
            print(latex_table)
        print(df)

    def store_results(self, output_path: str):
        SCORES = self._get_scores()
        df = pd.DataFrame(SCORES)
        df.to_csv(output_path, index=False)

    def store_mappings(self, output_path: str):
        '''
        Store both BERT and Comet mappings into a pickle file

        Args:
            output_path (str): Full path to the output pickle file
        '''
        # Create mapping dictionary to store
        mappings = {
            'bert_mapping': self.bert_mapping,
            'comet_mapping': self.comet_mapping
        }
        # Store mappings
        with open(output_path, 'wb') as f:
            pickle.dump(mappings, f)

        print(f"Mappings stored successfully in {output_path}")

    def load_mappings(self, input_path: str):
        '''
        Load both BERT and Comet mappings from a pickle file.

        Args:
            input_path: Full path to the input pickle file
        '''
        # Load mappings
        with open(input_path, 'rb') as f:
            mappings = pickle.load(f)

        # Restore mappings
        self.bert_mapping = mappings.get('bert_mapping', {})
        self.comet_mapping = mappings.get('comet_mapping', {})

        print(f"Mappings loaded successfully from {input_path}")
        print(
            f"Loaded BERT mappings for languages: {list(self.bert_mapping.keys())}")
        print(
            f"Loaded COMET mappings for languages: {list(self.comet_mapping.keys())}")


def create_matrix_from_csv(path_to_csv: str, metric: str = 'BLEU'):
    # Based on https://stackoverflow.com/questions/72827153/how-to-extract-specific-key-and-value-from-a-dataframe-python
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
    assert metric in df.columns, f"Metric '{metric}' not found in {path_to_csv}.\n Available metrics: {list(df.columns[1:])}"

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
