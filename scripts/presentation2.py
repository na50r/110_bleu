import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ORDER = ['da', 'sv', 'de', 'nl', 'en', 'es', 'fr', 'it', 'pt', 'el', 'fi']

def get_base_score_matrix(order=ORDER, zero_matrix=False):
    '''
    Returns Koehn's BLEU scores
    '''
    if zero_matrix:
        return pd.DataFrame(0, index=order, columns=order)
    data = {
        'da': [np.nan, 22.3, 22.7, 25.2, 24.1, 23.7, 20.0, 21.4, 20.5, 23.2, 30.3],
        'de': [18.4, np.nan, 17.4, 17.6, 18.2, 18.5, 14.5, 16.9, 18.3, 18.2, 18.9],
        'el': [21.1, 20.7, np.nan, 23.2, 28.3, 26.1, 18.2, 24.8, 17.4, 26.4, 22.8],
        'en': [28.5, 25.3, 27.2, np.nan, 30.5, 30.0, 21.8, 27.8, 23.0, 30.1, 30.2],
        'es': [26.4, 25.4, 31.2, 30.1, np.nan, 38.4, 21.1, 34.0, 22.9, 37.9, 28.6],
        'fr': [28.7, 27.7, 32.1, 31.1, 40.2, np.nan, 22.4, 36.0, 24.6, 39.0, 29.7],
        'fi': [14.2, 11.8, 11.4, 13.0, 12.5, 12.6, np.nan, 11.0, 10.3, 11.9, 15.3],
        'it': [22.2, 21.3, 26.8, 25.3, 32.3, 32.4, 18.3, np.nan, 20.0, 32.0, 23.9],
        'nl': [21.4, 23.4, 20.0, 21.0, 21.4, 21.1, 17.0, 20.0, np.nan, 20.2, 21.9],
        'pt': [24.3, 23.2, 27.6, 27.1, 35.9, 35.3, 19.1, 31.2, 20.7, np.nan, 25.9],
        'sv': [28.3, 20.5, 21.2, 24.8, 23.9, 22.6, 18.8, 20.2, 19.0, 21.9, np.nan]
    }

    languages = ['da', 'de', 'el', 'en', 'es',
                 'fr', 'fi', 'it', 'nl', 'pt', 'sv']
    base = pd.DataFrame(data, index=languages)

    base = base.reindex(index=order, columns=order)
    return base

def parse_results_from_folder(results_folder):
    '''
    Parses a folder of CSV files into a single dataframe.
    Format of CSV files:
    Label, BLEU, chrF, BERT-F1, COMET
    de-en, 30.69, 60.47, 80.23, 90.36
    
    Output dataframe format:
    BLEU, chrF, BERT-F1, COMET, dataset, translator, src_lang, tgt_lang
    30.69, 60.47, 80.23, 90.36, ep, deepl, de, en
    '''
    files = os.listdir(results_folder)
    dfs = []
    for fi in files:
        df = pd.read_csv(os.path.join(results_folder, fi))
        fi = fi.replace('.csv', '')
        dataset, translator = fi.split('-')
        df['dataset'] = dataset
        df['translator'] = translator
        df[['src_lang', 'tgt_lang']] = df['Label'].str.split('-', expand=True)
        del df['Label']
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df

def parse_results_from_file(results_file):
    df = pd.read_csv(results_file)
    df[['dataset', 'translator', 'src_lang', 'tgt_lang']] = df['Label'].str.split('-', expand=True)
    del df['Label']
    return df

### Correlations ###
def prepare_variable(df, metric, datasets, translators, src_lang=None, tgt_lang=None):
    subset = df[
        df['dataset'].isin(datasets) &
        df['translator'].isin(translators)
    ]
    if src_lang is not None:
        subset = subset[subset['src_lang'].isin(src_lang)]
    if tgt_lang is not None:
        subset = subset[subset['tgt_lang'].isin(tgt_lang)]
    subset = subset.rename(columns={metric: 'score'})
    subset = subset[['src_lang', 'tgt_lang', 'score', 'dataset', 'translator']]
    return subset


def show_correlations(df, config1, config2):
    df1 = prepare_variable(df, config1['metric'], config1['datasets'],
                           config1['translators'], config1['src_lang'], config1['tgt_lang'])
    df2 = prepare_variable(df, config2['metric'], config2['datasets'],
                           config2['translators'], config2['src_lang'], config2['tgt_lang'])
    merge_on = ['src_lang', 'tgt_lang']
    if len(config1['datasets']) > 1:
        merge_on.append('dataset')
    merged = df1.merge(df2, on=merge_on, suffixes=('_x', '_y'))

    pearson_corr, pearson_pval = pearsonr(merged['score_x'], merged['score_y'])
    spearman_corr, spearman_pval = spearmanr(
        merged['score_x'], merged['score_y'])
    print(f'Datasets: {config1["datasets"]} : {config2["datasets"]}')
    print(f'Translators: {config1["translators"]} : {config2["translators"]}')
    print(f'Metric: {config1["metric"]} : {config2["metric"]}')
    print(f"Pearson correlation: {pearson_corr:.2f} (p = {pearson_pval:.1e})")
    print(
        f"Spearman correlation: {spearman_corr:.2f} (p = {spearman_pval:.1e})")
    print()


### Score Matrix Operations ###
def form_matrix(df, metric, dataset, translator, order=ORDER):
    '''
    Forms a matrix from a dataframe.
    '''
    df = df[(df['dataset'] == dataset) & (df['translator'] == translator)]
    matrix =  df.pivot_table(
        index='src_lang',
        columns='tgt_lang',
        values=metric
    )
    return matrix.reindex(index=order, columns=order)

def plot_matrix(matrix, vmin=0, vmax=60, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, xlabel='Target Language', ylabel='Source Language', figsize=None, **kwargs):
    '''
    Displays score matrix in heatmap format
    '''
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=annot, fmt=fmt,
                cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yticks(rotation=0)
    plt.show()

def form_matrices(df, metric, translators, datasets, order=ORDER):
    '''
    Creates a dictionary of score matrices using translators and datasets as keys.
    '''
    assert len(translators) == len(
        datasets), 'Please provide same number of translators and datasets!'
    matrices = {}
    for tl, ds in zip(translators, datasets):
        matrix = form_matrix(df, metric, ds, tl, order)
        matrix = matrix.round(1)
        matrices[(ds, tl)] = matrix
    return matrices

def matrix_merger(matrices, merge_on):
    '''
    Averages out matrices based on a what to be merged on.
    '''
    assert merge_on in ['dataset', 'translator', 'all']
    datasets = [k[0] for k in matrices]
    translators = [k[1] for k in matrices]
    new = {}
    if merge_on == 'dataset':
        for d in set(datasets):
            selected = [m for k, m in matrices.items() if k[0] == d]
            new[d] = sum(selected) / len(selected)
            new[d] = new[d].round(1)
    elif merge_on == 'translator':
        for t in set(translators):
            selected = [m for k, m in matrices.items() if k[1] == t]
            new[t] = sum(selected) / len(selected)
            new[t] = new[t].round(1)
    elif merge_on == 'all':
        new['all'] = sum(matrices.values()) / len(matrices)
        new['all'] = new['all'].round(1)
    return new

def mode2mean(mode, key, matrix):
    '''
    Returns mean over rows, columns or difference between them.
    '''
    assert mode in ['INTO', 'FROM', 'DIFF']
    out = None
    if mode == 'INTO':
        # Mean over columns
        out = matrix.mean().round(1)
    elif mode == 'FROM':
        # Mean over rows
        out = matrix.mean(axis=1).round(1)
    elif mode == 'DIFF':
        out = matrix.mean(axis=1).round(1) - matrix.mean().round(1)

    out = out.reset_index()
    out.columns = ['lang', key]
    return out

def mode2vector(mode, key, lang, matrix):
    assert mode in ['INTO', 'FROM', 'DIFF']
    out = None
    if mode == 'INTO':
        # Mean over columns
        out = matrix[lang]
    elif mode == 'FROM':
        # Mean over rows
        out = matrix.loc[lang]
    elif mode == 'DIFF':
        out = matrix.loc[lang] - matrix[lang]

    out = out.reset_index()
    out.columns = ['lang', key]
    return out


def check_matrix(matrix):
    '''
    Confirms if matrix has 110 non-NaN values
    '''
    check = matrix.size - matrix.isna().sum().sum()
    return check == 110


def check_lang(matrix, lang, mode):
    '''
    Confirms if row, columns or both of chosen language has 10 non-NaN values
    '''
    if mode == 'INTO':
        return matrix[lang].count() == 10
    elif mode == 'FROM':
        return matrix.loc[lang].count() == 10
    elif mode == 'DIFF':
        return (matrix[lang].count() == 10) and (matrix.loc[lang].count() == 10)


def aggregate_matrices(matrices, mode='INTO', include_base=False):
    # Filter out matrices that don't have 110 non-NaN values
    matrices = {k: v for k, v in matrices.items() if check_matrix(v)}
    # Aggregate
    base = get_base_score_matrix(zero_matrix=(not include_base))
    dfs = []
    for key in matrices:
        label = '-'.join(key) if type(key) == tuple else key
        df = matrices[key] - base
        df = mode2mean(mode, label, df)
        dfs.append(df)
    base = mode2mean(mode, 'base', base)
    dfs.append(base)
    dfs = [df.set_index('lang') for df in dfs]
    comb = pd.concat(dfs, axis=1, join='inner')
    return comb


def extract_vectors(matrices, mode='INTO', lang='en', include_base=False):
    base = get_base_score_matrix(zero_matrix=(not include_base))
    dfs = []
    # Filter out matrices that don't have 10 non-NaN values in the chosen language
    matrices = {k: v for k, v in matrices.items() if check_lang(v, lang, mode)}
    for key in matrices:
        label = '-'.join(key) if type(key) == tuple else key
        df = matrices[key] - base
        df = mode2vector(mode, label, lang, df)
        dfs.append(df)
    base = mode2vector(mode, 'base', lang, base)
    dfs.append(base)
    dfs = [df.set_index('lang') for df in dfs]
    comb = pd.concat(dfs, axis=1, join='inner')
    comb = comb.drop(lang)
    return comb


def plot_vectors(df, label_map={}, color_map={}, include_base=False, linestyle=None):
    langs = df.index
    x = range(len(langs))

    if include_base:
        plt.plot(
            x, df['base'],
            marker='o',
            label=label_map.get('base', 'base'),
            linestyle=linestyle,
            color=color_map.get('base', None)
        )
    for col in df.columns:
        if col == 'base':
            continue
        plt.plot(
            x, df[col] + df['base'],
            marker='o',
            label=label_map.get(col, col),
            linestyle=linestyle,
            color=color_map.get(col, None)
        )
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=x, labels=langs)
    plt.show()
