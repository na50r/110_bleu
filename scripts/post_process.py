# NOTE: This code is still work in progress!
import os
import json
from os.path import join, exists
import unicodedata


def full_normalize(text):
    return unicodedata.normalize('NFKC', text.replace('\xa0', ' ')).strip()

def direct_triplet_align(mt_sents: list[str], ref_sents: list[str], src_sents: list[str], folder_path: str, filename: str):
    '''
    Aligns source, reference and machine translation in COMET format directly
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''
    align_cnt = 0
    norm = full_normalize
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(join(folder_path, f'{filename}.jsonl'), 'w') as f:
        for mt, ref, src in zip(mt_sents, ref_sents, src_sents):
            if mt and ref and src:
                obj = {'mt':norm(mt), 'ref':norm(ref), 'src':norm(src)}
                align_cnt+=1
                print(json.dumps(obj), file=f)
    return align_cnt


def align_sents(src_sents: list[str], tgt_sents: list[str], folder_path: str, src_lang: str, tgt_lang: str, model: str = 'paraphrase-multilingual-MiniLM-L12-v2', filename=None, is_split=False):
    '''
    Uses bertalign to align source and target sentences
    Uses paraphrase-multilingual-MiniLM-L12-v2 as default sentence embedding model
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

    if filename is None:
        filename = f'{src_lang}-{tgt_lang}.jsonl'
    else:
        filename = f'{filename}.jsonl'

    out_file = join(folder_path, filename)
    if exists(out_file):
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
        model=model,
        is_split=is_split
    )
    aligner.align_sents()
    src_sents_a, tgt_sents_a = aligner.get_sents()

    with open(out_file, 'w') as f:
        for s, t in zip(src_sents_a, tgt_sents_a):
            o = {'src': s, 'tgt': t}
            print(json.dumps(o), file=f)


def post_triplet_align(src_sents_org: list[str], src_sents_ali: list[str], ref_sents_org: list[str], mt_sents_ali: list[str], folder_path: str, filename: str):
    '''
    Alignes re-aligned source, reference and machine translation in COMET format
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''
    if not exists(folder_path):
        os.makedirs(folder_path)
    aligned_cnt = 0
    out_file = f'{filename}.jsonl'
    norm = full_normalize
    src2mt = {norm(s.strip()): norm(m.strip())
              for s, m in zip(src_sents_ali, mt_sents_ali)}
    src2ref = {norm(s.strip()): norm(r.strip())
               for s, r in zip(src_sents_org, ref_sents_org)}

    check = frozenset(src2ref.keys())
    keys = [k for k in src2mt if k in check]
    discared = []
    with open(join(folder_path, out_file), 'w') as f:
        for k in keys:
            m = src2mt[k]
            r = src2ref[k]
            if m and k and r:
                aligned_cnt += 1
                o = {'mt': m, 'ref': r, 'src': k}
                print(json.dumps(o), file=f)
            else:
                discared.append({'mt': m, 'ref': r, 'src': k})
    return aligned_cnt, discared


def load_sents_from_file(filename: str, folder: str) -> list[str]:
    filename = f'{filename}.txt'
    file_path = join(folder, filename)
    with open(file_path, 'r') as f:
        sents = [s.strip() for s in f]
    return sents


def load_aligned_sents_from_file(filename: str, folder: str) -> tuple[list[str], list[str]]:
    file_path = join(folder, f'{filename}.jsonl')

    src_sents, tgt_sents = [], []
    with open(file_path, 'r') as f:
        for ln in f:
            o = json.loads(ln)
            src_sents.append(o['src'])
            tgt_sents.append(o['tgt'])

    return src_sents, tgt_sents
