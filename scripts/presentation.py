import os
from os.path import join
from scripts.scoring import create_matrix_from_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scripts.util import LANG_ISO
from collections import defaultdict
import pandas as pd
import matplotlib.colors as mcolors

def lang_freq(labels):
    freq_dict = defaultdict(int)
    for l in labels:
        from_l, to_l = l.split('-')
        freq_dict[f'to-{to_l}'] += 1
        freq_dict[f'from-{from_l}'] += 1
    return {key: v for (key, v) in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)}

COLORS = {
    "ep": "#d690ff",
    "opus": "#3e2249",
    "flores": "#7c367a",
    "deepl": "#db1919",
    "gpt": "#10C221"
}

TWO_COLORS = {
    'ep-deepl': "#D83838",
    'ep-gpt': "#39C04B",
    'flores-deepl': "#981133",
    'flores-gpt': "#247C31",
    'opus-deepl': "#530A0A",
    'opus-gpt': "#0C3A1B"
}

DATASETS = {"ep", "opus", "flores"}
TRANSLATORS = {"deepl", "gpt"}


def mix_colors(*hex_colors):
    rgbs = [mcolors.to_rgb(c) for c in hex_colors]
    avg_rgb = tuple(sum(ch) / len(ch) for ch in zip(*rgbs))
    return mcolors.to_hex(avg_rgb)


def get_label_color(label: str):
    matched = [k for k in COLORS if k in label]
    matched_set = set(matched)

    if not matched:
        return "#000000"  

    if len(matched) == 1:
        return COLORS[matched[0]]
    
    check_set = DATASETS.union(TRANSLATORS)
    if len(matched_set.intersection(check_set)) == 2:
        return TWO_COLORS[f'-'.join(matched)]

    is_dataset = matched_set <= DATASETS
    is_translator = matched_set <= TRANSLATORS

    if is_dataset or is_translator:
        return mix_colors(*[COLORS[m] for m in matched])

    return mix_colors(*[COLORS[m] for m in matched])

def get_colors(labels):
    return {label: get_label_color(label) for label in labels}

def get_koehn(show=False):
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

    languages = ['da', 'de', 'el', 'en', 'es', 'fr', 'fi', 'it', 'nl', 'pt', 'sv']
    koehn_2005 = pd.DataFrame(data, index=languages)
    
    order = ['da', 'sv', 'de', 'nl', 'en', 'es', 'fr', 'it', 'pt', 'el', 'fi']
    koehn_2005 = koehn_2005.reindex(index=order, columns=order)
    
    if show:
        sns.heatmap(koehn_2005, annot=True, fmt=".1f",
                    cmap="YlGnBu", cbar=True, vmin=0, vmax=60)
        plt.yticks(rotation=0)
        plt.show()
    return koehn_2005


class Presenter:
    # Linguistically sensible order
    ORDER = ['da', 'sv', 'de', 'nl', 'en', 'es', 'fr', 'it', 'pt', 'el', 'fi']
    SCORE_MATRIX = {
        'BLEU': {'vmin': 0, 'vmax': 60}
    }

    def __init__(self,
                 results_folder,
                 metrics=['BLEU'],
                 order=ORDER
                 ):
        '''
        Args:
            results_folder: Path to folder containing CSV files with name dataset_translator.csv of format:
            Label, Metrics
            
            Example:
            Label, BLEU, chrF, COMET
            de-en, 30.69, 60.47, 90.36
        '''
        self.store = results_folder
        self.metrics = metrics
        self.order = order
        self.datasets = set([fn.split('-')[0]
                            for fn in os.listdir(results_folder)])
        self.translators = set([fn.split('-')[1].replace('.csv', '')
                                for fn in os.listdir(results_folder)])
        self.cases = {f.replace('.csv', ''): {'file': join(
            results_folder, f)} for f in os.listdir(results_folder)}
        for k in self.cases:
            file_path = self.cases[k]['file']
            for m in metrics:
                df = create_matrix_from_csv(file_path, metric=m)
                df = df.reindex(index=order, columns=order)
                self.cases[k][m] = df


    def show_score_matrices(self, metric='BLEU', focus=None, with_koehn = False, merge= None, **heatmap_kwargs):
        '''
        Displays score matrices as in Europarl paper with heatmap formatting
        '''
        data = self._prepare_data(metric, focus)
        dfs = {}
        if with_koehn:
            dfs['koehn'] = get_koehn()
               
        if merge is not None:
            data = self._merge_data(data, merge, metric)
            data = {k: v[metric] for k, v in data.items()}
            dfs.update(data)
        else:
            data = {k: v[metric] for k, v in data.items()}
            dfs.update(data)
            
        for k in dfs:
            df = dfs[k]
            df = df.round(1)
            print(k)
            kwargs = self.SCORE_MATRIX.get(metric, {})
            kwargs.update(heatmap_kwargs)
            sns.heatmap(df, annot=True, fmt=".1f",
                        cmap="YlGnBu", cbar=True, **kwargs)
            plt.yticks(rotation=0)
            plt.show()
            print()


    def _validate_config(self, config):
        assert config[
            'metric'] in self.metrics, f'Please provide a metric that is within {self.metrics}!'
        assert set(config['datasets']).issubset(
            self.datasets), f'Please provide datasets that are within {self.datasets}!'
        assert set(config['translators']).issubset(
            self.translators), f'Please provide translators that are within {self.translators}!'
        assert config.get('src_lang', None) is None or set(config['src_lang']).issubset(
            self.order), f'Please provide source languages that are within {self.order}!'
        assert config.get('tgt_lang', None) is None or set(config['tgt_lang']).issubset(
            self.order), f'Please provide target languages that are within {self.order}!'

    def _validate_configs(self, config1, config2):
        assert (len(config1['datasets']) == 1 or config1['datasets'] == config2['datasets']
                ), 'If you provide multiple datasets, make sure they are in same order!'


    def prepare_variable(self, config):
        '''
        Transforms score matrix of datasets and translator into a more configurable dataframe for visualizations
        
        config = {
            'datasets' : [],
            'translators : [],
            'src_lang' : [] | None,
            'tgt_lang' : [] | None,
            'metric' : ''
        }
        
        New format:
        dataset, translator, src_lang, tgt_lang, score
        '''
        self._validate_config(config)
        dfs = []
        metric = config['metric']
        for d in config['datasets']:
            for t in config['translators']:
                key = f'{d}-{t}'
                df = self.cases[key][metric]
                melted = df.reset_index()
                melted = melted.melt(
                    id_vars=melted.columns[0], var_name='tgt_lang', value_name='score')
                melted.columns = ['src_lang', 'tgt_lang', 'score']
                melted['dataset'] = d
                melted['translator'] = t
                dfs.append(melted)
        df = pd.concat(dfs, ignore_index=True)
        update = []
        if config.get('src_lang', None):
            df_src = df[df['src_lang'].isin(config['src_lang'])]
            update.append(df_src)
        if config.get('tgt_lang', None):
            df_tgt = df[df['tgt_lang'].isin(config['tgt_lang'])]
            update.append(df_tgt)
        if len(update) > 0:
            df = pd.concat(update, ignore_index=True)
            return df.dropna()
        return df.dropna()
    
    def show_correlations(self, config1, config2):
        self._validate_configs(config1, config2)
        df1 = self.prepare_variable(config1)
        df2 = self.prepare_variable(config2)
        merge_on = ['src_lang', 'tgt_lang']
        if len(config1['datasets']) > 1:
            merge_on.append('dataset')
        merged = df1.merge(df2, on=merge_on, suffixes=('_x', '_y'))
        
        pearson_corr, pearson_pval = pearsonr(merged['score_x'], merged['score_y'])
        spearman_corr, spearman_pval = spearmanr(merged['score_x'], merged['score_y'])
        print(f'Datasets: {config1["datasets"]} : {config2["datasets"]}')
        print(f'Translators: {config1["translators"]} : {config2["translators"]}')
        print(f"Pearson correlation: {pearson_corr:.2f} (p = {pearson_pval:.1e})")
        print(
            f"Spearman correlation: {spearman_corr:.2f} (p = {spearman_pval:.1e})")
        print()


    def linear_regression(self, config1, config2, x_label, y_label, color_by=None, custom_color=None, plot=True):
        self._validate_configs(config1, config2)
        df1 = self.prepare_variable(config1)
        df2 = self.prepare_variable(config2)
        merge_on = ['src_lang', 'tgt_lang']
        if len(config1['datasets']) > 1:
            merge_on.append('dataset')
        merged = df1.merge(df2, on=merge_on, suffixes=('_x', '_y'))
        model = np.polyfit(merged['score_x'], merged['score_y'], 1)
        
        if plot==True:
            x_line = np.linspace(merged['score_x'].min(),
                                    merged['score_x'].max(), 100)
            y_line = model[0] * x_line + model[1]
            color_params = {'hue':color_by}
            
            if custom_color:
                tag_color, palette = self._customize_color(custom_color)
                merged['marked'] = merged.apply(tag_color, axis=1)
                color_params['hue'] = 'marked'
                color_params['palette'] = palette
            
            sns.scatterplot(
                data=merged,
                x='score_x',
                y='score_y',
                **color_params
            )
            plt.plot(x_line, y_line, color='black')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()
        return merged, model
        
    def top_n_residuals(self, data, model, top_n=20):
        data = data.copy()  
        data['predicted_y'] = np.poly1d(model)(data['score_x'])
        data['residual'] = abs(data['score_y'] - data['predicted_y'])
        top_n_entries = data.sort_values(by='residual', ascending=False).head(top_n)
        return top_n_entries
        
    def _customize_color(self, custom_color):
        palette = {}
        for key in custom_color:
            if key == 'src_lang':
                for lang in custom_color[key]:
                    lbl = f'From {LANG_ISO[lang]}'
                    palette.update({lbl:custom_color[key][lang]})
            elif key == 'tgt_lang':
                for lang in custom_color[key]:
                    lbl = f'Into {LANG_ISO[lang]}'
                    palette.update({lbl:custom_color[key][lang]})
            else:
                for k in custom_color[key]:
                    palette.update({k.upper():custom_color[key][k]})
        palette.update({'Other':'gray'})
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


    ### Aggregation ###
    # Refactored with help of Claude Sonnet 4
    # Original code had a lot of duplication, asked it to extract common parts
    def _validate_params(self, mode, merge, focus, with_koehn, metric):
        """Validate common parameters used by both metric functions."""
        assert mode in ['INTO', 'FROM', 'DIFF']
        assert (merge is None) or (merge in ['DATASET', 'TRANSLATOR', 'ALL'])
        # assert (focus is None and merge is None) or (focus is None and merge is not None) or (
        #     focus is not None and merge is None), 'Merge and Focus should not be used together!'
        assert (with_koehn == True and metric ==
                'BLEU') or with_koehn == False, 'Use with_koehn only with BLEU scores!'

    def _prepare_data(self, metric, focus, exclude_opus=False):
        """Prepare data by filtering cases and applying focus if specified."""
        if exclude_opus:
            data = {k: {metric: v[metric]}
                        for k, v in self.cases.items() if not k.startswith('opus')}
        else:
            data = {k: {metric: v[metric]}
                    for k, v in self.cases.items()}

        if focus is not None:
            subset = set(focus)
            keys = set(data.keys())
            assert subset.intersection(keys) == subset

            new = {k: {} for k in focus}
            for key in focus:
                new[key][metric] = data[key][metric]
            data = new
        return data

    def _merge_data(self, data, merge, metric) -> dict[str, dict[str, pd.DataFrame]]:
        """Merge data across datasets or translators."""
        if merge is None:
            return data
              
        datasets = [key.split('-')[0] for key in data]
        translators = [key.split('-')[1] for key in data]

        if merge == 'DATASET':
            new = {d: {metric: []} for d in set(datasets)}
            def func(x, y): return x.startswith(y)
        elif merge == 'TRANSLATOR':
            new = {t: {metric: []} for t in set(translators)}
            def func(x, y): return x.endswith(y)
        elif merge == 'ALL':
            new = {'all': {metric: []}}
            def func(x, y): return True

        # Collect dataframes to be merged
        for k in new:
            for key in data:
                if func(key, k):
                    df = data[key][metric]
                    new[k][metric].append(df)
        
        # Average dataframes
        for k in new:
            dfs = new[k][metric]
            df = sum(dfs) / len(dfs)
            new[k][metric] = df
        return new

    def _get_base_data(self, with_koehn, data, metric):
        """Get base data (Koehn scores or zeros)."""
        if with_koehn:
            base = get_koehn()
        else:
            k = list(data.keys())[0]
            sample = data[k][metric]
            base = pd.DataFrame(0, index=sample.index, columns=sample.columns)
        return base

    def _create_plot(self, comb, base_indexed, mode, metric, with_koehn, colors, title, xlabel, ylabel, lang=None):
        """Create the plot with common styling and layout."""
        langs = base_indexed.index
        x = range(len(langs))

        if mode in ['INTO', 'FROM']:
            linestyle = None
        else:
            linestyle = 'None'

        if with_koehn:
            plt.plot(x, base_indexed['koehn'], marker='o',
                     label='koehn', linestyle=linestyle)

        for col in comb.columns:
            if col == 'koehn':
                continue
            if col.endswith('deepl'):
                marker = 's'
            else:
                marker = '^'

            color = colors.get(col, None)
            plt.plot(x, comb[col] + base_indexed['koehn'],
                     marker=marker, label=col, color=color, linestyle=linestyle)

        plt.xticks(ticks=x, labels=langs)

        # Set default titles
        if lang is not None:
            if mode == 'INTO':
                _title = f'{metric} Scores for Translations INTO {LANG_ISO[lang]}'
            elif mode == 'FROM':
                _title = f'{metric} Scores for Translations FROM {LANG_ISO[lang]}'
            elif mode == 'DIFF':
                _title = f'{metric} From/Into Differences'
        else:
            if mode == 'INTO':
                _title = f'Mean {metric} Scores for Translations INTO Language'
            elif mode == 'FROM':
                _title = f'Mean {metric} Scores for Translations FROM Language'
            elif mode == 'DIFF':
                _title = f'Mean {metric} From/Into Differences'

        title = title or _title
        xlabel = xlabel or 'Language'
        if mode in ['FROM', 'INTO']:
            ylabel = ylabel or f'{metric} score'
        else:
            ylabel = ylabel or f'Mean {metric} differences'

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def mean_metric_from_or_into_lang(self, mode='INTO', plot=True, with_koehn=True, metric='BLEU', title=None, xlabel=None, ylabel=None, colors=None, merge=None, focus=None):
        '''
        Plots Mean Metric Scores with/without Koehn's scores and produces a pandas.DataFrame containing mean scores or differences if Koehn's scores are used.
        '''
        self._validate_params(mode, merge, focus, with_koehn, metric)


        data = self._prepare_data(metric, focus, exclude_opus=True)
        data = self._merge_data(data, merge, metric)
        _colors = get_colors(data.keys())
        colors = colors or _colors
        base = self._get_base_data(with_koehn, data, metric)

        diffs = {}
        for key in data:
            df = data[key][metric]

            if mode == 'INTO':
                diff = df.mean().round(1) - base.mean().round(1)
            elif mode == 'FROM':
                diff = df.mean(axis=1).round(1) - base.mean(axis=1).round(1)
            elif mode == 'DIFF':
                diff = df.mean(axis=1) - df.mean()
                diff = diff.round(1)

            diff = diff.reset_index()
            diff.columns = ['lang', key]
            diffs[key] = diff

        if mode == 'INTO':
            base_series = base.mean().round(1)
        elif mode == 'FROM':
            base_series = base.mean(axis=1).round(1)
        elif mode == 'DIFF':
            base_series = base.mean(axis=1) - base.mean()
            base_series = base_series.round(1)

        base_df = base_series.reset_index()
        base_df.columns = ['lang', 'koehn']
        base_indexed = base_df.set_index('lang')

        if with_koehn:
            dfs = [base_df]
        else:
            dfs = []
        for key in diffs:
            dfs.append(diffs[key])

        dfs = [df.set_index('lang') for df in dfs]
        comb = pd.concat(dfs, axis=1, join='inner')

        # Plotting
        if plot == True:
            self._create_plot(comb, base_indexed, mode, metric,
                              with_koehn, colors, title, xlabel, ylabel)

        return comb

    def metric_from_or_into_language(self, mode='INTO', plot=True, with_koehn=True, metric='BLEU', title=None, xlabel=None, ylabel=None, colors=None, merge=None, focus=None, lang='en'):
        self._validate_params(mode, merge, focus, with_koehn, metric)

        data = self._prepare_data(metric, focus, exclude_opus=lang != 'en')
        _colors = get_colors(data.keys())
        data = self._merge_data(data, merge, metric)
        colors = colors or _colors
        base = self._get_base_data(with_koehn, data, metric)

        diffs = {}
        for key in data:
            df = data[key][metric]

            if mode == 'INTO':
                diff = df[lang].round(1) - base[lang]
            elif mode == 'FROM':
                diff = df.loc[lang].round(1) - base.loc[lang]
            elif mode == 'DIFF':
                diff = df.loc[lang] - df[lang]
                diff = diff.round(1)

            diff = diff.reset_index()
            diff.columns = ['lang', key]
            diffs[key] = diff

        if mode == 'INTO':
            base_series = base[lang]
        elif mode == 'FROM':
            base_series = base.loc[lang]
        elif mode == 'DIFF':
            base_series = base.loc[lang] - base[lang]

        base_df = base_series.reset_index()
        base_df.columns = ['lang', 'koehn']

        if with_koehn:
            dfs = [base_df]
        else:
            dfs = []
        for key in diffs:
            dfs.append(diffs[key])

        dfs = [df.set_index('lang') for df in dfs]
        comb = pd.concat(dfs, axis=1, join='inner')
        comb = comb.drop(lang)
        base_indexed = base_df.set_index('lang')
        base_indexed = base_indexed.drop(lang)

        if plot == True:
            self._create_plot(comb, base_indexed, mode, metric,
                              with_koehn, colors, title, xlabel, ylabel, lang)
        return comb
