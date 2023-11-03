# tutorial : https://datascience.stackexchange.com/questions/10695/how-to-initialize-a-new-word2vec-model-with-pre-trained-model-weights
# for ft: https://towardsdatascience.com/using-fasttext-and-svd-to-visualise-word-embeddings-instantly-5b8fa870c3d1
import codecs

import os
import sys
sys.path.append(os.path.abspath("scr_py"))

import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

from setup import prepare_docs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


df = pd.read_json('data/df_chat_socio.json')
df_prep, df = prepare_docs(df, X="Chat_subject", y="honest1000", dv="declared_income_final") # "Chat_group_all"


##################
# own embeddings #
##################

### w2v
w2v_own = Word2Vec(df.doc_words, vector_size=300, min_count=2)
my_list = list(w2v_own.wv.index_to_key) # creating a list with my vocabulary to use this for the pretrained embeddings

### ft
command = './fastText/fasttext skipgram -input fastText/chat.pre.txt -output fastText/model'
os.system(command)

words = []
vecs = []
with codecs.open('./fastText/model.vec', 'r', 'utf-8') as f_in:
    # Skip the header line if it exists (common in .vec files)
    next(f_in)
    for line in f_in:
        word, vec = line.strip().split(' ', 1)
        words.append(word)
        vecs.append(np.fromstring(vec, sep=' '))

indices = [words.index(w) for w in my_list if w in words]
reduced_word_vectors = [vecs[i] for i in indices]
common_elements = list(set(my_list).intersection(words))
twodim = PCA().fit_transform(reduced_word_vectors)[:, :2]



##########################
# pre-trained embeddings #
##########################
# download .txt from https://www.deepset.ai/german-word-embeddings

# pre: word2vec model
model_w2v = KeyedVectors.load_word2vec_format("data/w2v_size.txt", binary=False, encoding='utf-8')  #vocab size and vector length as numbers in the first row -> use add_vec_size.py
new_key_to_index = {key[2:-1]: val for key, val in model_w2v.key_to_index.items()}
model_w2v.key_to_index = new_key_to_index
model_w2v.index_to_key = list(new_key_to_index.keys())
model_list = list(model_w2v.index_to_key)
common_elements_w2v = list(set(my_list).intersection(set(model_list)))


# pre: glove model
model_glove = KeyedVectors.load_word2vec_format('data/glove.txt', binary=False, no_header=True)
glove_list = model_glove.index_to_key
glove_list_lower = [word.lower() for word in glove_list]
common_elements_glove = list(set(my_list).intersection(set(glove_list_lower)))


def display_pca_scatterplot(model, ax, words=None, sample=0, words_to_label=None, title=""):
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    if words_to_label is None:
        words_to_label = words

    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:, :2]

    ax.scatter(twodim[:, 0], twodim[:, 1], c='grey', s=2, alpha=0.2)
    for word, (x, y) in zip(words, twodim):
        if word in words_to_label:
            ax.scatter(x, y, c='blue', s=5)
            ax.text(x, y, word)
    ax.set_title(title)


words_to_label = ["risiko", "einkommen", "strafe"]
fig, axs = plt.subplots(1, 4, figsize=(6.71, 6.71 / 4))

# fastText
axs[0].scatter(twodim[:, 0], twodim[:, 1], c='grey', s=2, alpha=0.2)
for word, (x, y) in zip(common_elements, twodim):
    if word in words_to_label:
        axs[0].scatter(x, y, c='blue', s=5)
        axs[0].text(x, y, word)
axs[0].set_title("fastText, own")

# all other
display_pca_scatterplot(w2v_own.wv, axs[1], words=my_list, words_to_label=words_to_label, title="Word2Vec, own")
display_pca_scatterplot(model_w2v, axs[2], words=common_elements_w2v, words_to_label=words_to_label, title="Word2Vec, pre")
display_pca_scatterplot(model_glove, axs[3], words=common_elements_glove, words_to_label=words_to_label, title="GloVe, pre")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.subplots_adjust(wspace=0.00005)
plt.tight_layout()
plt.savefig('figures/embed_combined_adjusted.pdf')
plt.show()
