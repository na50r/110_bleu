import numpy as np
import json
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

def compute_bert_score(ref, hyp, lang, rescale_with_baseline=True, model_type=None):
    # Recale with Baseline set to True as specified in: https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
    from bert_score import score
    P, R, F1 = score(hyp, ref, lang=lang, verbose=False,
                     rescale_with_baseline=rescale_with_baseline, model_type=model_type)
    return F1


class ResultProducer:
    def __init__(self, label2files: dict[str, str] = None, use_bert: bool = False, use_comet: bool = False, bert_rescale: bool = False, bert_model_type: str=None):
        '''
        Args:
            label2files: A dictionary that uses labels as keys and filepath to a JSONL file of COMET format {"mt":"sent", "ref":"sent", "src":"sent"}
            use_bert: A boolean, decide if you want to compute BERT-F1 score or not, not recommended for CPU
            use_comet: A boolean, decide if you want to compute COMET score or not, not recommended for CPU
        '''
        self.use_bert = use_bert
        self.use_comet = use_comet
        self.label2files = label2files
        if self.use_comet:
            self.comet_model = load_comet_model()
        self.bert_rescale = bert_rescale
        self.bert_model_type = bert_model_type
        
        
    def compute_result(self, label: str, file_path: str):
        tgt_lang = label.split('-')[-1]
        with open(file_path, 'r') as f:
            data = [json.loads(ln) for ln in f]

        mt_sents = [o['mt'] for o in data]
        ref_sents = [o['ref'] for o in data]
        
        bleu_score = compute_bleu(ref_sents, mt_sents)
        chrf_score = compute_chrf(ref_sents, mt_sents)
        bert_f1_score = None
        comet_score = None
        
        if self.use_bert == True:
            start = time.time()
            bert_out = compute_bert_score(ref_sents, mt_sents, lang=tgt_lang, rescale_with_baseline=self.bert_rescale, model_type=self.bert_model_type)
            end = time.time()
            logging.info(f'BERTScore took {end-start:.2f} seconds for {label}')
            # BERT-F1 score computed as value between 0 to 1
            # To make results cleaner, we adhere to BLEU & chrF and make it value between 0 and 100
            bert_f1_score = np.array(bert_out).mean() * 100
        
        if self.use_comet == True:
            start = time.time()
            comet_out = compute_comet(data, self.comet_model)
            end = time.time()
            logging.info(
                f'COMET took {end-start:.2f} seconds for {label}')
            # COMET score computed as value between 0 to 1
            # To make results cleaner, we adhere to BLEU & chrF and make it value between 0 and 100
            comet_score = comet_out['system_score']*100
        return bleu_score, chrf_score, bert_f1_score, comet_score
    
    def _create_header(self, output_path: str):
        header = ['Label', 'BLEU', 'chrF']
        if self.use_bert == True:
            header.append('BERT-F1')
        if self.use_comet == True:
            header.append('COMET')
        with open(output_path, 'w') as f:
            f.write(','.join(header) + '\n')
            
    def _append_row(self, output_path: str, label: str, bleu_score: float, chrf_score: float, bert_f1_score: float, comet_score: float):
        row = [label, bleu_score, chrf_score]
        if self.use_bert == True:
            row.append(bert_f1_score)
        if self.use_comet == True:
            row.append(comet_score)
        with open(output_path, 'a') as f:
            row = [str(r) for r in row]
            f.write(','.join(row) + '\n')
    
    def compute_and_store_results(self, output_path: str):
        self._create_header(output_path)
        computed = []
        for label in self.label2files:
            file_path = self.label2files[label]
            try:
                computed.append(label)
                bleu_score, chrf_score, bert_f1_score, comet_score = self.compute_result(label, file_path)
                self._append_row(output_path, label, bleu_score, chrf_score, bert_f1_score, comet_score)
                logging.info(f'[✔️]: Scored for {label} {len(computed)}/{len(self.label2files)}')
            except Exception as e:
                logging.error(f'[🔥]: Error {str(e)}')
                logging.info(f'[ℹ️]: Computed for {computed}')
                logging.debug("Traceback:", exc_info=True)
