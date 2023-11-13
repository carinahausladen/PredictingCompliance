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


df = pd.read_csv('../data/df_chat_socio.csv')
df_prep, df = prepare_docs(df, y="honestmean", X="Chat_subject", dv="declared_income_final")

df.doc_words = [["kein", "chat"] if not doc else doc for doc in df.doc_words] #sometimes there is an empty [] when chat was only stop words
df.new_docs = [["kein chat"] if not doc else doc for doc in df.doc_words]

df.new_docs = [" ".join(doc) if isinstance(doc, list) else doc for doc in df.new_docs] # Ensure that all elements in df.new_docs are strings
df.new_docs = [doc if isinstance(doc, str) and doc.strip() != "" else "kein chat" for doc in df.new_docs]

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
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format("../data/w2v_size.txt", binary=False) #, unicode_errors='ignore')
# important about averaged embeddings https://stackoverflow.com/questions/65121932/how-to-use-deepsets-word-embedding-pre-trained-models-using-gensim
new_key_to_index = {key[2:-1]: val for key, val in w2v.key_to_index.items()}
w2v.key_to_index = new_key_to_index


def avg_embdngs(documents, embeddings, num_features):
    oov_vector = np.zeros(num_features) # default vector for OOV words

    vectors = []
    for doc in tqdm(documents):
        words = word_tokenize(doc)
        word_embeddings = []
        for word in words:
            if word in embeddings.key_to_index:  # Check if the word is in the model's vocabulary
                word_embeddings.append(embeddings[word])
            else:
                # Use the default OOV vector
                word_embeddings.append(oov_vector)
              #  print(f"OOV word: {word}")

        if word_embeddings:
            vectors.append(np.mean(word_embeddings, axis=0)) # calc mean of embeddings
        else:
            vectors.append(oov_vector)
            print(f"Failed on document: {doc}")

    return np.array(vectors)


num_features = w2v.vector_size
df.new_docs = [x if len(x) != 0 else "kein chat" for x in df.new_docs]
X_w2v_pre = avg_embdngs(df.new_docs, w2v, num_features)

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

def tfidf_embdngs(documents, embeddings):
    vectors = []
    for doc in tqdm(documents):
        words = word_tokenize(doc)
        word_vectors_and_weights = [(embeddings[word], idf_dict.get(word, 1)) for word in words if word in embeddings]

        if word_vectors_and_weights:  # Check if there is at least one word vector
            word_vectors, weights = zip(*word_vectors_and_weights)  # Unzips into two lists
            word_vectors = np.array(word_vectors)
            weights = np.array(weights)
            weighted_average = np.average(word_vectors, axis=0, weights=weights)
            vectors.append(weighted_average)
        else:
            # Append zero vector if no words found in the embeddings
            vectors.append(np.zeros(embeddings.vector_size))
            print(f"Failed on document: {doc}")

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
glove = KeyedVectors.load_word2vec_format('../data/glove.txt', binary=False, no_header=True)
glove_list = glove.index_to_key
glove_list_lower = [word.lower() for word in glove_list]
num_features=glove.vector_size

X_glove_pre = avg_embdngs(df.new_docs, glove, num_features)

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
# ft uses Logistic Regression and sentence embeddings; metrics caluclated and safed in class_fast.py
m_ft_own = np.load("../data/m_ft_own.npy") # these are my own trained

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
df_results.round(decimals=3).to_latex(buf="../figures/embdgs_over_mean.tex")
print(df_results.to_latex(float_format="{:0.1f}".format))


