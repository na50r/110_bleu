# NOTE: This code is still work in progress!
import os
import json
from os.path import join, exists


def direct_triplet_align(mt_sents: list[str], ref_sents: list[str], src_sents: list[str], src_lang: str, ref_lang: str, folder_path: str):
    '''
    Aligns source, reference and machine translation in COMET format directly
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(join(folder_path, f'{src_lang}-{ref_lang}.jsonl'), 'a') as f:
        for mt, ref, src in zip(mt_sents, ref_sents, src_sents):
            obj = dict()
            obj['mt'] = mt
            obj['ref'] = ref
            obj['src'] = src
            print(json.dumps(obj), file=f)


def align_sents(src_sents : list[str], tgt_sents : list[str], src_lang : str, tgt_lang : str, folder_path : str) -> tuple[list[str], list[str]]:
    '''
    Uses bertalign to align source and target sentences
    Uses paraphrase-multilingual-MiniLM-L12-v2 as sentence embedding model
    Args:
        src_sents: list of source sentences
        tgt_sents: list of target sentences
        src_lang: ISO code of source language
        tgt_lang: ISO code of target language
        folder_path: path to folder where aligned sentences should be stored
    Returns:
        aligned source and target sentences
    '''
    if not exists(folder_path):
        os.makedirs(folder_path)

    mt_filename = f'{src_lang}-{tgt_lang}.{tgt_lang}'
    src_filename = f'{src_lang}-{tgt_lang}.{src_lang}'

    mt_file = join(folder_path, mt_filename)
    src_file = join(folder_path, src_filename)
    if exists(mt_file) and exists(src_file):
        print(f'{src_lang} and {tgt_lang} already aligned!')
        return

    from bertalign import Bertalign
    tgt_text = '\n'.join(tgt_sents)
    src_text = '\n'.join(src_sents)
    aligner = Bertalign(
        src=src_text,
        tgt=tgt_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model='paraphrase-multilingual-MiniLM-L12-v2'
    )
    aligner.align_sents()
    src_sents_a, tgt_sents_a = aligner.get_sents()

    with open(join(folder_path, src_filename), 'w') as f:
        for sent in src_sents_a:
            print(sent, file=f)

    with open(join(folder_path, mt_filename), 'w') as f:
        for sent in tgt_sents_a:
            print(sent, file=f)

    return src_sents_a, tgt_sents_a


def post_triplet_align(src_sents_org : list[str], src_sents_ali: list[str], ref_sents_org: list[str], mt_sents_ali: list[str], src_lang: str, ref_lang: str, folder_path: str):
    '''
    Alignes re-aligned source, reference and machine translation in COMET format
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''
    if not exists(folder_path):
        os.makedirs(folder_path)
    aligned_cnt = 0
    out_file = f'{src_lang}-{ref_lang}.jsonl'
    with open(join(folder_path, out_file), 'w') as f:
        for x, sent in enumerate(src_sents_org):
            for y, src_sent in enumerate(src_sents_ali):
                if sent.strip() == src_sent.strip():
                    obj = dict()
                    obj['src'] = sent.strip()
                    obj['ref'] = ref_sents_org[x].strip()
                    obj['mt'] = mt_sents_ali[y].strip()
                    if obj['src'] != '' and obj['ref'] != '' and obj['mt'] != '':
                        print(json.dumps(obj), file=f)
                        aligned_cnt += 1
                else:
                    continue
    print(f"{aligned_cnt} sents aligned for {src_lang} and {ref_lang}")

def load_mt_sents(dataset: str, translator: str, src_lang: str, tgt_lang: str) -> list[str]:
    filename = f'{dataset}-{translator}-{src_lang}-{tgt_lang}.txt'
    file_path = join('translations', filename)
    with open(file_path, 'r') as f:
        mt_sents = [s.strip() for s in f.readlines()]
    return mt_sents
