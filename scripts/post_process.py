# NOTE: This code is still work in progress!
import os
import json
from os.path import join, exists


def direct_triplet_align(mt_sents: list[str], ref_sents: list[str], src_sents: list[str], src_lang: str, ref_lang: str, folder_path):
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


def align_src_mt_sents(src_sents, mt_sents, src_lang, mt_lang, folder_path):
    '''
    Aligns source and machine translation in case of misalignments in output
    '''
    if not exists(folder_path):
        os.makedirs(folder_path)

    mt_filename = f'{src_lang}-{mt_lang}.{mt_lang}'
    src_filename = f'{src_lang}-{mt_lang}.{src_lang}'

    mt_file = join(folder_path, mt_filename)
    src_file = join(folder_path, src_filename)
    if exists(mt_file) and exists(src_file):
        print(f'{src_lang} and {mt_lang} already aligned!')
        return

    from bertalign import Bertalign
    mt_text = '\n'.join(mt_sents)
    src_text = '\n'.join(src_sents)
    aligner = Bertalign(
        src=src_text,
        tgt=mt_text,
        src_lang=src_lang,
        tgt_lang=mt_lang,
        model='paraphrase-multilingual-MiniLM-L12-v2'
    )
    aligner.align_sents()
    src_sents_a, mt_sents_a = aligner.get_sents()

    with open(join(folder_path, src_filename), 'w') as f:
        for sent in src_sents_a:
            print(sent, file=f)

    with open(join(folder_path, mt_filename), 'w') as f:
        for sent in mt_sents_a:
            print(sent, file=f)

    return src_sents_a, mt_sents_a


def post_triplet_align(src_sents_org, src_sents_ali, ref_sents_org, mt_sents_ali, src_lang, ref_lang, folder_path):
    '''
    Alings source, reference and machine translation in COMET format after source and machine translation were correctly aligned
    Uses src_sents as a key to align ref_sents and mt_sents with each other, missing sentences are discarded in this process
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
