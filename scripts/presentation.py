import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scripts.util import LANG_ISO


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
    '''
    Parses a folder of CSV files into a single dataframe.
    Format of CSV files:
    Label, BLEU, chrF, BERT-F1, COMET
    de-en, 30.69, 60.47, 80.23, 90.36
    '''
    df = pd.read_csv(results_file)
    df[['dataset', 'translator', 'src_lang', 'tgt_lang']
       ] = df['Label'].str.split('-', expand=True)
    del df['Label']
    return df

### NOTE ###
# All the following functions apply operations on a pandas.DataFrame of format:
# BLEU, chrF, BERT-F1, COMET, dataset, translator, src_lang, tgt_lang

### METRIC CORRELATIONS AND LINEAR REGRESSION ###


def prepare_variable(df, metric, datasets, translators, src_lang=None, tgt_lang=None):
    '''
    Selects the relevant rows from the data based on metric, datasets, translators, src_lang, and tgt_lang.
    '''
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


def correlations(df, config1, config2, show=False):
    '''
    Excepts two configs of format:
    config = {
        'datasets': ['ep', 'flores'],
        'translators': ['deepl', 'gpt'],
        'src_lang': None,
        'tgt_lang': None,
        'metric': 'BLEU'
    }
    Computes the Pearson and Spearman correlations between selected data based on the configs.
    '''
    df1 = prepare_variable(df, **config1)
    df2 = prepare_variable(df, **config2)
    merge_on = ['src_lang', 'tgt_lang']
    if len(config1['datasets']) > 1:
        merge_on.append('dataset')
    merged = df1.merge(df2, on=merge_on, suffixes=('_x', '_y'))

    pearson_corr, pearson_pval = pearsonr(merged['score_x'], merged['score_y'])
    spearman_corr, spearman_pval = spearmanr(
        merged['score_x'], merged['score_y'])
    if show:
        print(f'Datasets: {config1["datasets"]} : {config2["datasets"]}')
        print(
            f'Translators: {config1["translators"]} : {config2["translators"]}')
        print(f'Metric: {config1["metric"]} : {config2["metric"]}')
        print(
            f"Pearson correlation: {pearson_corr:.2f} (p = {pearson_pval:.1e})")
        print(
            f"Spearman correlation: {spearman_corr:.2f} (p = {spearman_pval:.1e})")
    return pearson_corr, pearson_pval, spearman_corr, spearman_pval


def linear_regression(df, config1, config2, x_label, y_label, color_by=None, custom_color=None, label_map=None, plot=True):
    '''
    Expects two configs of format:
    config = {
        'datasets': ['ep', 'flores'],
        'translators': ['deepl', 'gpt'],
        'src_lang': None,
        'tgt_lang': None,
        'metric': 'BLEU'
    }
    Plots a linear regression between selected data based on the configs.
    If custom_colors are desired, requires format:
    custom_color = {
        'src_lang': {'da': 'red', 'de': 'blue'},
        'tgt_lang': {'da': 'green', 'de': 'yellow'}
        'dataset': {'ep': 'purple', 'flores': 'orange'}
    }
    '''
    df1 = prepare_variable(df, **config1)
    df2 = prepare_variable(df, **config2)
    merge_on = ['src_lang', 'tgt_lang']

    if len(config1['datasets']) > 1:
        merge_on.append('dataset')

    merged = df1.merge(df2, on=merge_on, suffixes=('_x', '_y'))
    model = np.polyfit(merged['score_x'], merged['score_y'], 1)

    if plot:
        x_line = np.linspace(merged['score_x'].min(),
                             merged['score_x'].max(), 100)
        y_line = model[0] * x_line + model[1]
        color_params = {'hue': color_by}
        if custom_color:
            merged, palette = set_custom_colors(merged, custom_color)
            color_params['palette'] = palette
            color_params['hue'] = 'marked'

        sns.scatterplot(
            data=merged,
            x='score_x',
            y='score_y',
            **color_params
        )
        plt.plot(x_line, y_line, color='black')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if label_map:
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = [label_map.get(label, label) for label in labels]
            plt.legend(handles, new_labels, title=color_params['hue'])
        plt.show()
    return merged, model


def customize_color(custom_color):
    '''
    Prepares colors and legend matching those colors
    For src_lang/tgt_lang, 'src_lang: ['da', 'de'] becomes 'From Danish' and 'From German'
    '''
    palette = {}
    for key in custom_color:
        if key == 'src_lang':
            for lang in custom_color[key]:
                label = f'From {LANG_ISO[lang]}'
                palette.update({label: custom_color[key][lang]})
        elif key == 'tgt_lang':
            for lang in custom_color[key]:
                label = f'Into {LANG_ISO[lang]}'
                palette.update({label: custom_color[key][lang]})
        else:
            for k in custom_color[key]:
                palette.update({k.upper(): custom_color[key][k]})
    palette.update({'Other': 'gray'})

    def tag_color(row):
        for key in custom_color:
            if row[key] in custom_color[key]:
                if key == 'src_lang':
                    return f'From {LANG_ISO[row[key]]}'
                elif key == 'tgt_lang':
                    return f'Into {LANG_ISO[row[key]]}'
                else:
                    return row[key].upper()
        return 'Other'
    return tag_color, palette


def set_custom_colors(df, custom_color):
    tag_color, palette = customize_color(custom_color)
    df['marked'] = df.apply(tag_color, axis=1)
    '''
    df after tag_color applied (Example):
    src_lang, tgt_lang, score_x, dataset_x, translator_x, score_y, dataset_y, translator_y, marked
    sv, da, 36.07..., ep, deepl, 61.52..., ep, deepl, Other
    de, da, 34.97..., ep, deepl, 60.55..., ep, deepl, From German
    ...

    palette = {
        'Other': 'gray',
        'From German': 'green',
    }
    '''
    return df, palette


def top_n_residuals(data, model, top_n=20):
    data = data.copy()
    data['predicted_y'] = np.poly1d(model)(data['score_x'])
    data['residual'] = abs(data['score_y'] - data['predicted_y'])
    top_n_entries = data.sort_values(
        by='residual', ascending=False).head(top_n)
    return top_n_entries


def mark_residual_by_src_or_tgt_freq(outliers, n=4):
    '''
    Computes the top n source and target languages based on the outliers
    and returns a custom color dictionary for those languages.
    '''
    src_lang_cnts = outliers['src_lang'].value_counts().to_dict()
    tgt_lang_cnts = outliers['tgt_lang'].value_counts().to_dict()
    tmp1 = {f'src-{k}': v for k, v in src_lang_cnts.items()}
    tmp2 = {f'tgt-{k}': v for k, v in tgt_lang_cnts.items()}
    tmp = {**tmp1, **tmp2}
    top = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:n]
    out = {'src_lang': {}, 'tgt_lang': {}}
    # ChatGPT Aided
    palette = sns.color_palette("YlGnBu", n)
    cols = palette[::-1]
    for i, (k, v) in enumerate(top):
        if k.startswith('src'):
            out['src_lang'][k[4:]] = cols[i]
        else:
            out['tgt_lang'][k[4:]] = cols[i]
    return out, top

### SCORE MATRIX OPERATIONS ###


def form_matrix(df, metric, dataset, translator, order=ORDER):
    '''
    Forms a matrix from a dataframe.
    '''
    df = df[(df['dataset'] == dataset) & (df['translator'] == translator)]
    matrix = df.pivot_table(
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
    '''
    Selects the row, column or difference between them for a given language.
    '''
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
    '''
    Given a dictionary of score matrices, aggregates them into a dataframe consisting of the mean over rows, mean over columns or mean difference between them.
    Note: This dataframe can be considered as a 'list of vectors'
    '''
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
    '''
    Given a dictionary of score matrices, extracts the row, column or difference between them for a given language.
    Note: This dataframe can be considered as a 'list of vectors'
    '''
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


def plot_vectors(vectors, label_map={}, color_map={}, include_base=False, linestyle=None, ylabel=None, legend=plt.legend):
    langs = vectors.index
    x = range(len(langs))

    if include_base:
        plt.plot(
            x, vectors['base'],
            marker='o',
            label=label_map.get('base', 'base'),
            linestyle=linestyle,
            color=color_map.get('base', None)
        )
    for col in vectors.columns:
        if col == 'base':
            continue
        plt.plot(
            x, vectors[col] + vectors['base'],
            marker='o',
            label=label_map.get(col, col),
            linestyle=linestyle,
            color=color_map.get(col, None)
        )
    legend()
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True)
    plt.xticks(ticks=x, labels=langs)
    plt.show()
