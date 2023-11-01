'table results from loop'
import pandas as pd
import pandas as pd


def dict_to_df_new(d):
    rows = []
    for first_key, second_dict in d.items():
        for second_key, third_dict in second_dict.items():
            for third_key, metrics in third_dict.items():
                row = metrics.copy()
                row['data'] = first_key
                row['y_variation'] = second_key
                row['x_variation'] = third_key
                rows.append(row)
    return pd.DataFrame(rows)

def one_pickle_to_df(path_to_pickle):
    results = pd.read_pickle(path_to_pickle)

    data_frames = []
    for n in results:
        data_frames.append(dict_to_df_new(n))

    scores_df = pd.concat(data_frames, ignore_index=True)
    return scores_df


results_2 = one_pickle_to_df('data/df_y_x.pickle')
performance_metrics = ['f1score', 'precision', 'recall', 'AUC', 'accuracy']

idx = results_2.groupby('data')['accuracy'].idxmax()
max_data = results_2.loc[idx, ['data'] + performance_metrics]
max_data = max_data.rename(columns={'data': 'Variable'})

idx = results_2.groupby('y_variation')['AUC'].idxmax()
max_y_variation = results_2.loc[idx, ['y_variation'] + performance_metrics]
max_y_variation = max_y_variation.rename(columns={'y_variation': 'Variable'})

idx = results_2.groupby('x_variation')['f1score'].idxmax()
max_x_variation = results_2.loc[idx, ['x_variation'] + performance_metrics]
max_x_variation = max_x_variation.rename(columns={'x_variation': 'Variable'})

final_results = pd.concat([max_data, max_y_variation, max_x_variation]).round(3)
latex_table = final_results.to_latex(index=False, formatters={'f1score': "{:.3f}".format, 'precision': "{:.3f}".format, 'recall': "{:.3f}".format, 'AUC': "{:.3f}".format, 'accuracy': "{:.3f}".format})
print(latex_table)

