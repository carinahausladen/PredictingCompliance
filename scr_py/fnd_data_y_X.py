# Source: https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b
# https://github.com/TomLin/Playground/blob/master/04-Model-Comparison-Word2vec-Doc2vec-TfIdfWeighted.ipynb
# glove from here https://deepset.ai/german-word-embeddings German Wikipedia

# seems not to run in ipython console
'finds the best combination of data X and y'

import os
script_dir = '/Users/carinah/Documents/GitHub/PredictingCompliance/scr_py'
os.chdir(script_dir)

import sys
sys.path.append(os.path.abspath("scr_py"))

import itertools
import multiprocessing as mp
import os
import pickle
import time
import warnings

import gensim
import pandas as pd
import spacy
from gensim.models.word2vec import Word2Vec
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from UtilWordEmbedding import DocPreprocess
from setup import prepare_X_y
from strt_grp_sffl_splt import str_grp_splt
from utility import fit_n_times, adjusted_f1

##############################################################

ros = RandomOverSampler(random_state=42)
warnings.simplefilter('ignore')
pd.set_option('max_colwidth', 1000)
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise."
workers = mp.cpu_count()

nlp = spacy.load("de_core_news_sm")
#nlp = spacy.load('de')  # .venv/bin/python -m spacy download de
stop_words = spacy.lang.de.stop_words.STOP_WORDS
##############################################################

# read and prep df
df = pd.read_csv('../data/df_chat_socio.csv')  # 855 rows, 9 columns
df_spllchckd = pd.read_csv('../data/df_chat_socio_splchckd.csv')  # 4815 rows, 46 cols

df = prepare_X_y(df, dv="declared_income_final")
df_spllchckd = prepare_X_y(df_spllchckd, dv="declared_income_final")  # I need to run prepare df over the spellchecked one

# define loop vars
df_vars = {
    "duplicated": df,
    "spell_checked": df_spllchckd
}

y_vars = {
     "<1000": 'honest1000',
    "<500": 'honest500',
    "<mean": 'honestmean',
}

X_vars = {
    "chat_subject": 'Chat_subject',
    "chat_group": 'Chat_group_all',
   "chat_group_label": 'Chat_group_label',
    "label_group": 'Tags',
}


#val_X = "Chat_subject"
#val_y = "honest500"


def prepare_feat(data, df_vars, y_vars, X_vars):
    name_df, df = df_vars
    name_y, val_y = y_vars
    name_X, val_X = X_vars

    start = time.time()
    print(name_df, name_y, name_X)

    if name_df == "duplicated":
        df = data
    else:
        df = data.drop_duplicates()

    # prepare X
    if not name_X == "label_group":
        df_all_docs = DocPreprocess(nlp, stop_words, df[val_X], df[val_y])
        print("finished DocPreprocess")

        tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)  # vectorize bf split!
        tfidf_X = tfidf.fit_transform(df_all_docs.new_docs)
    else:
        # val_X = "Tags"
        X_raw = df[val_X]

        tfidf = TfidfVectorizer()
        tfidf_X = tfidf.fit_transform(X_raw.values.astype('U')).toarray()

    # get indices
    train_idx, test_idx = str_grp_splt(df,
                                       grp_col_name="Group_ID_simuliert",
                                       y_col_name=val_y,
                                       train_share=0.8)
    print("finished split")

    if not name_X == "label_group":
        # prepare train/test X, y
        train_X = tfidf_X[train_idx]
        test_X = tfidf_X[test_idx]

        train_y = df_all_docs.labels[train_idx]
        test_y = df_all_docs.labels[test_idx]

        train_X, train_y = ros.fit_resample(train_X, train_y)  # oversample minority
    else:
        train_X = tfidf_X[train_idx]
        test_X = tfidf_X[test_idx]

        train_y = df[val_y][train_idx]
        test_y = df[val_y][test_idx]

        train_X, train_y = ros.fit_resample(train_X, train_y)  # oversample minority

    # prepare dict
    scores = dict()
    scores[name_df] = dict()
    scores[name_df][name_y] = dict()
    scores[name_df][name_y][name_X] = dict()

    # clf
    print("start gridsearch")
    svm = SVC(probability=True)
    svm_params = {'C': [10 ** (x) for x in range(-1, 4)],
                  'kernel': ['poly', 'rbf', 'linear'],
                  'degree': [2, 3]}
    score = make_scorer(adjusted_f1, greater_is_better=True, needs_proba=True)

    grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, scoring=score, verbose=0, refit=False)
    grid.fit(train_X, train_y)
    best_params = grid.best_params_
    print("finished gridsearch")
    svm = SVC(C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'],
              probability=True)  # refit with best params
    metrics_svm = fit_n_times(svm, train_X, train_y, test_X, test_y)

    scores[name_df][name_y][name_X] = dict()
    scores[name_df][name_y][name_X]["f1score"] = metrics_svm[0]
    scores[name_df][name_y][name_X]["precision"] = metrics_svm[1]
    scores[name_df][name_y][name_X]["recall"] = metrics_svm[2]
    scores[name_df][name_y][name_X]["AUC"] = metrics_svm[3]
    scores[name_df][name_y][name_X]["accuracy"] = metrics_svm[4]

    stop = time.time()
    duration = stop - start
    print(duration)

    return scores


# prep loop
results = []
jobs = list(itertools.product(*[df_vars.items(), y_vars.items(), X_vars.items()]))
print(len(jobs))

###########################
# parallel version of loop#
###########################
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn')
    pool = mp.Pool(10)
    results = [pool.apply(prepare_feat, args=(df, d, y, x)) for d, y, x in jobs]
    pool.close()


with open('../data/df_y_x.pickle', 'wb') as fp:
    pickle.dump(results, fp)