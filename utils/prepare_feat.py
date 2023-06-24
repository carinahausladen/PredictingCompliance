import time

import spacy
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .UtilWordEmbedding import DocPreprocess
from .strt_grp_sffl_splt import str_grp_splt
from .utility import fit_n_times, adjusted_f1

nlp = spacy.load("de_core_news_sm")
stop_words = spacy.lang.de.stop_words.STOP_WORDS

ros = RandomOverSampler(random_state=42)

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
    df_all_docs = DocPreprocess(nlp, stop_words, df[val_X], df[val_y])
    print("finished DocPreprocess")

    tfidf = TfidfVectorizer(input='content', 
                            lowercase=False, 
                            preprocessor=lambda x: x)  # vectorize bf split!
    tfidf_X = tfidf.fit_transform(df_all_docs.new_docs)

    # get indices
    train_idx, test_idx = str_grp_splt(df,
                                       grp_col_name="group",
                                       y_col_name=val_y,
                                       train_share=0.8)
    print("finished split")

    # prepare train/test X, y
    train_X = tfidf_X[train_idx]
    test_X = tfidf_X[test_idx]

    train_y = df_all_docs.labels[train_idx]
    test_y = df_all_docs.labels[test_idx]

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

    grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1, 
                        scoring=score, verbose=0, refit=False)
    grid.fit(train_X, train_y)
    best_params = grid.best_params_

    print("finished gridsearch")
    svm = SVC(C=best_params['C'], 
              kernel=best_params['kernel'], 
              degree=best_params['degree'],
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

