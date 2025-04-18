from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

#MODEL, EMBED_DIM, NUM_OF_LANGS
MODEL_384_50 = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_384_94 = "multilingual-e5-small"
MODEL_512_50 = "distiluse-base-multilingual-cased-v2"
MODEL_768_110 = "LaBSE"

class CosineSim:
    def __init__(self, model_name=MODEL_512_50):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Encode list of sentences into list of embeddings
    def encode(self, sents: list[str]):
        embeddings = self.model.encode(sents)
        # Convert to tensor right away
        return torch.tensor(embeddings, dtype=torch.float32)

    # Compute similarities of all embeddings
    # Compute mean, max and min
    def similarity(self, src_sents, tgt_sents, metrics=True):
        src_embs = self.encode(src_sents)
        tgt_embs = self.encode(tgt_sents)

        if len(src_embs) != len(tgt_embs):
            raise ValueError(
                "Src and Tgt embedding array should have same length!")

        sim_scores = self.cos(src_embs, tgt_embs)

        if metrics:
            mean = torch.mean(sim_scores)
            max_sim = torch.max(sim_scores)
            min_sim = torch.min(sim_scores)
            print(f'Mean {mean:.2f}')
            print(f'Max {max_sim:.2f}')
            print(f'Min {min_sim:.2f}')
            return mean, max_sim, min_sim

        return sim_scores
    

def cosine_sim_matrix(jsonlfile, langs):
    import json
    import pandas as pd
    with open(jsonlfile, 'r') as f:
        data = [json.loads(ln) for ln in f]
    compute_time = sum(o['time'] for o in data)
    
    lang2dict = {lang:{} for lang in langs}
    for lang in lang2dict:
        for o in data:
            for lang in o['pair']:
                if o['pair'][0] == lang:
                    other = o['pair'][1]
                else:
                    other = o['pair'][0]
                lang2dict[lang][other] = o['sim']
    
    dfs = []
    for lang in lang2dict:
        df = pd.DataFrame(
            list(lang2dict[lang].items()),
            columns=['Src', 'Sim']
        )
        dfs.append((lang, df))
    
    records = []
    for df in dfs:
        for _, row in df[1].iterrows():
            records.append({
                'Src' : row['Src'],
                'Tgt' : df[0],
                'Sim' : row['Sim']
            })
    
    all_sims = pd.DataFrame(records)

    sim_matrix = all_sims.pivot(
        index='Src', columns='Tgt', values='Sim'
    ).round(2)
    
    sim_matrix = all_sims.pivot(
        index='Src', columns='Tgt', values='Sim'
    ).round(2)
    
    return sim_matrix, compute_time
