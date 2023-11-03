'calculates embeddings'
# LINK: https://towardsdatascience.com/nlp-performance-of-different-word-embeddings-on-text-classification-de648c6262b

import os
import sys
sys.path.append(os.path.abspath("scr_py"))
import numpy as np

import multiprocessing

import pandas as pd
from gensim.models.word2vec import Word2Vec
from imblearn.over_sampling import RandomOverSampler
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from UtilWordEmbedding import DocModel
from UtilWordEmbedding import MeanEmbeddingVectorizer
from UtilWordEmbedding import TfidfEmbeddingVectorizer
from setup import prepare_docs
from strt_grp_sffl_splt import str_grp_splt
from utility import run_log_reg
from pymagnitude import *


df = pd.read_csv('data/df_chat_socio.csv')
df_prep, df = prepare_docs(df, y="honestmean", X="Chat_subject", dv="declared_income_final")
ros = RandomOverSampler(random_state=42, sampling_strategy='minority')  # oversampling!
df_prep["honestmean"].value_counts()  # 1: minority = compliance

##############
# embeddings #
##############
print('bow')
bow = CountVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)
X_bow = bow.fit_transform(df.new_docs)
train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = X_bow[train_idx]
test_X = X_bow[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

train_y.value_counts()
test_y.value_counts()

m_bow, model, ma = run_log_reg(train_X, test_X, train_y, test_y)

# tfidf
print('tfidf')
tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)
X_tfidf = tfidf.fit_transform(df.new_docs)
train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = X_tfidf[train_idx]
test_X = X_tfidf[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_tfidf = run_log_reg(train_X, test_X, train_y, test_y)

#######
# w2v #
#######
print('w2v')
# own, simple
w2v_own = Word2Vec(df.doc_words, vector_size=70, min_count=1)
mean_vec_tr = MeanEmbeddingVectorizer(w2v_own)
doc_vec = mean_vec_tr.transform(df.doc_words)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honestmean",
                                   train_share=0.8)
train_X = doc_vec[train_idx]
test_X = doc_vec[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)
test_y.value_counts()

m_w2v_own_smpl = run_log_reg(train_X, test_X, train_y, test_y)

# own, tfidf
tfidf_vec_tr = TfidfEmbeddingVectorizer(w2v_own)
tfidf_vec_tr.fit(df.doc_words)
tfidf_doc_vec = tfidf_vec_tr.transform(df.doc_words)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = tfidf_doc_vec[train_idx]
test_X = tfidf_doc_vec[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_w2v_own_tfidf = run_log_reg(train_X, test_X, train_y, test_y)

# pre, simple
# python -m pymagnitude.converter -i 'analysis/data/vectors_w2v.txt' -o 'analysis/data/w2v.magnitude'
w2v = Magnitude('data/w2v.magnitude')


def avg_embdngs(documents, embedings, num_trials=10):
    vectors = []
    for title in tqdm(documents):
        try:
            emb = np.average(embedings.query(word_tokenize(title)), axis=0)
            vectors.append(emb)
        except:
            print(f"Failed")
            print(title)
    return np.array(vectors)


df.new_docs = [x if len(x) != 0 else "kein_chat" for x in df.new_docs]
X_w2v_pre = avg_embdngs(df.new_docs, w2v)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = X_w2v_pre[train_idx]
test_X = X_w2v_pre[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_w2v_pre_smpl = run_log_reg(train_X, test_X, train_y, test_y)

# pre, idf
tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)
tfidf.fit(df.new_docs)
idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))


def tfidf_embdngs(documents, embedings):
    vectors = []
    for title in tqdm(documents):
        w2v_vectors = embedings.query(word_tokenize(title))
        weights = [idf_dict.get(word, 1) for word in word_tokenize(title)]
        vectors.append(np.average(w2v_vectors, axis=0, weights=weights))
    return np.array(vectors)


X_tfidf_w2v_pre = tfidf_embdngs(df.new_docs, w2v)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = X_tfidf_w2v_pre[train_idx]
test_X = X_tfidf_w2v_pre[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_w2v_pre_tfidf = run_log_reg(train_X, test_X, train_y, test_y)

########
# glove#
########
# glove, pre
# convert file before!
glove = Magnitude('data/glove_vec.magnitude')
X_glove_pre = avg_embdngs(df.new_docs, glove)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honestmean",
                                   train_share=0.8)
train_X = X_glove_pre[train_idx]
test_X = X_glove_pre[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_glove_pre_smpl = run_log_reg(train_X, test_X, train_y, test_y)

# pre, idf
tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)
tfidf.fit(df.new_docs)
idf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

X_tfidf_glove_pre = tfidf_embdngs(df.new_docs, glove)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honestmean",
                                   train_share=0.8)
train_X = X_tfidf_glove_pre[train_idx]
test_X = X_tfidf_glove_pre[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_glove_pre_tfidf = run_log_reg(train_X, test_X, train_y, test_y)

####
# ft#
####

# ft, own
'ft uses LR and sentence embeddings; txt from class_fast.py'
m_ft_own = np.load("data/fasttext_embeddings.npy")

##########
# do2vec #
##########
workers = multiprocessing.cpu_count()
dm_args = {'dm': 1, 'dm_mean': 1, 'vector_size': 100, 'window': 5, 'negative': 5, 'hs': 0, 'min_count': 2,
           'sample': 0, 'workers': workers, 'alpha': 0.025, 'min_alpha': 0.025, 'epochs': 100,
           'comment': 'alpha=0.025'
           }
dm = DocModel(docs=df.tagdocs, **dm_args)
dm.custom_train()
dm_doc_vec_ls = []
for i in range(len(dm.model.dv)):
    dm_doc_vec_ls.append(dm.model.dv[i])
dm_doc_vec = pd.DataFrame(dm_doc_vec_ls)

train_idx, test_idx = str_grp_splt(df_prep,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honestmean",
                                   train_share=0.8)
train_X = dm_doc_vec.loc[train_idx]
test_X = dm_doc_vec.loc[test_idx]
train_y = df_prep["honestmean"][train_idx]
test_y = df_prep["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

m_dm_doc_vec = run_log_reg(train_X, test_X, train_y, test_y)

###################
# table results ###
###################

lst = [list(m_bow), list(m_tfidf[0]),
       list(m_w2v_own_smpl[0]), list(m_w2v_own_tfidf[0]),
       list(m_w2v_pre_smpl[0]), list(m_w2v_pre_tfidf[0]),
       list(m_glove_pre_smpl[0]), list(m_glove_pre_tfidf[0]),
       list(m_ft_own), list(m_dm_doc_vec[0])]

df_results = pd.DataFrame(lst, columns=['f1score', 'precision', 'recall', 'AUC', 'accuracy'], dtype=float)
df_results=df_results*100
df_results.rename(index={0: 'bag of words', 1: 'bag of words (tf-idf)',
                         2: 'Word2Vec (own, avg)', 3: 'Word2Vec (own, tf-idf)',
                         4: 'Word2Vec (pre, avg)', 5: 'Word2Vec (pre, tf-idf)',
                         6: 'GloVe (pre, avg)', 7: 'GloVe (pre, tf-idf)',
                         8: 'fastText (own, avg)', 9: 'Doc2Vec'}, inplace=True)
df_results = df_results.sort_values(by='f1score', ascending=False)
df_results.round(decimals=3).to_latex(buf="figures/embdgs_over_mean.tex")
print(df_results.to_latex(float_format="{:0.1f}".format))


