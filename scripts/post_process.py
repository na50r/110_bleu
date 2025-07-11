import os
import json
from os.path import join, exists

def normalize(text: str) -> str:
    # Remove non-breaking space for FLORES+ French
    # Only for string-matching, should not be stored
    return text.replace('\xa0', ' ').strip()

def direct_triplet_align(mt_sents: list[str], ref_sents: list[str], src_sents: list[str], folder_path: str, filename: str) -> int:
    '''
    Aligns source, reference and machine translation in COMET format directly
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''
    align_cnt = 0
    norm = normalize
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(join(folder_path, f'{filename}.jsonl'), 'w') as f:
        for mt, ref, src in zip(mt_sents, ref_sents, src_sents):
            if norm(mt) and norm(ref) and norm(src):
                obj = {'mt':mt, 'ref':ref, 'src':src}
                align_cnt+=1
                print(json.dumps(obj), file=f)
    return align_cnt


def align_sents(src_sents: list[str], tgt_sents: list[str], folder_path: str, src_lang: str, tgt_lang: str, model: str = 'paraphrase-multilingual-MiniLM-L12-v2', filename: str=None, is_split: bool=False, fix_side=None):
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
        is_split=is_split,
        fix_side=fix_side
    )
    aligner.align_sents()
    src_sents_a, tgt_sents_a = aligner.get_sents()

    with open(out_file, 'w') as f:
        for s, t in zip(src_sents_a, tgt_sents_a):
            o = {'src': s, 'tgt': t}
            print(json.dumps(o), file=f)


def post_triplet_align(src_sents_org: list[str], src_sents_ali: list[str], ref_sents_org: list[str], mt_sents_ali: list[str], folder_path: str, filename: str) -> tuple[int, list[dict[str]]]:
    '''
    Alignes re-aligned source, reference and machine translation in COMET format
    Assumes that mt_sents, ref_sents and src_sents are aligned with each other
    '''
    if not exists(folder_path):
        os.makedirs(folder_path)
    aligned_cnt = 0
    out_file = f'{filename}.jsonl'
    norm = normalize
    norm2src = {norm(s): s for s in src_sents_org}
    src2mt = {norm(s): m
              for s, m in zip(src_sents_ali, mt_sents_ali)}
    src2ref = {norm(s): r
               for s, r in zip(src_sents_org, ref_sents_org)}

    check = set(src2ref.keys())
    keys = [k for k in src2mt if k in check]
    discarded = []
    with open(join(folder_path, out_file), 'w') as f:
        for k in keys:
            s = norm2src[k]
            m = src2mt[k]
            r = src2ref[k]
            if m and k and r:
                aligned_cnt += 1
                o = {'mt': m, 'ref': r, 'src': s}
                print(json.dumps(o), file=f)
            else:
                discarded.append({'mt': m, 'ref': r, 'src': s})
    return aligned_cnt, discarded


def load_sents_from_file(filename: str, folder: str) -> list[str]:
    filename = f'{filename}.txt'
    file_path = join(folder, filename)
    with open(file_path, 'r') as f:
        sents = [s.strip() for s in f]
    return sents


def load_aligned_sents_from_file(filename: str, folder: str, src_label='src', tgt_label='tgt') -> tuple[list[str], list[str]]:
    file_path = join(folder, f'{filename}.jsonl')

    src_sents, tgt_sents = [], []
    with open(file_path, 'r') as f:
        for ln in f:
            o = json.loads(ln)
            src_sents.append(o[src_label])
            tgt_sents.append(o[tgt_label])
    return src_sents, tgt_sents
