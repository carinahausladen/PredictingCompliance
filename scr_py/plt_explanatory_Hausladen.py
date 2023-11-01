import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import spacy
import gensim
import numpy as np
#from .UtilWordEmbedding import DocPreprocess

def prepare_X_y(df, dv):
    # prepare x vars
    chat_cols = ['Chat_subject', 'Chat_group_all', 'Chat_sel']
    df.loc[:, chat_cols] = df.loc[:, chat_cols].fillna('kein_chat').astype(str)
    df.loc[:, chat_cols] = df.loc[:, chat_cols].applymap(lambda x: x if x != "" else "kein_chat")

    # generate y vars; dv=declared_income
    df['honest10'] = (df[dv] <= 10).astype(int)
    df['honest30'] = (df[dv] < 30).astype(int)
    df['honestmean'] = (df[dv] < df[dv].mean()).astype(int)

    def define_classes(x):
        if x == 10:
            return 1
        elif x == 60:
            return 0
        else:
            return 2

    df['honest3label'] = df[dv].apply(lambda x: define_classes(x))

    return df
def prepare_docs(df, y, X, dv):
    df = prepare_X_y(df, dv)

    pattern = '|'.join(["XD", "xd", "xD",
                        "X-D", "x-d", "x-D",
                        ":D", ";D",
                        ":-D", ";-D",
                        ":\)", ";\)",
                        ":-\)", ";-\)", "haha"
                        ])
    df.loc[:, X] = df.loc[:, X].str.replace(pattern, "smiley", regex=True)

    nlp = spacy.load("de_core_news_sm")  # .venv/bin/python -m spacy download de
    stop_words = spacy.lang.de.stop_words.STOP_WORDS
    all_docs = DocPreprocess(nlp, stop_words, df[X], df[y])

    return df, all_docs

df_chat_hours = pd.read_csv('data/chat_hours.csv')
df_prep_new, all_docs_new = prepare_docs(df_chat_hours,
                                         y="honest10", X="Chat_subject", dv="player.hours_stated")


########### PLOT HISTOGRAM ##############
df = pd.read_csv('data/df_chat_hours.csv')
df[['player.hours_stated']].hist(bins=20, grid=False)
plt.title("")
plt.xlabel('hours stated', fontsize=18)
plt.ylabel('frequency', fontsize=16)
plt.savefig('figures/fig4b.pdf')
plt.show()

######################### add stacked barcharts
plot_df = pd.DataFrame()
plot_df["10"] = df_prep_new["honest10"].value_counts()
plot_df["30"] = df_prep_new["honest30"].value_counts()
plot_df["mean"] = df_prep_new["honestmean"].value_counts()
plot_df = plot_df.rename(index={1: "honest",
                                2: "undefined",
                                0: "dishonest"})

plot_df.T.plot(kind="bar", stacked=True)
plt.legend(loc='best')  # , bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.xlabel('threshold to binarize reported income', fontsize=18)
plt.ylabel('frequency', fontsize=16)
plt.savefig('figures/fig4b.pdf')
plt.show()


############ Distribtuion of three indicators
from nltk import word_tokenize
import seaborn as sns

all_docs = all_docs_new
df = df_prep_new
df = df.rename(columns={"honest10": "honest1000"})
df.columns
s = df.Chat_group_all.str.len().sort_values().index

dishonest = [len(word_tokenize(Chat_group_all)) for Chat_group_all in
             df[df.honest1000.values == 0].Chat_group_all.values]
honest = [len(word_tokenize(Chat_group_all)) for Chat_group_all in df[df.honest1000.values != 0].Chat_group_all.values]

plt.figure(figsize=(5,4))
sns.distplot(dishonest, bins=10, label='dishonest')
sns.distplot(honest, bins=10, label='honest')
plt.legend()
plt.xlabel('Number of Words')
plt.tight_layout()
plt.savefig('figures/fig5a.pdf')
plt.show()

def mean_word_length(x):
    word_lengths = np.array([])
    for word in word_tokenize(x):
        word_lengths = np.append(word_lengths, len(word))
    return word_lengths.mean()

dishonest_len = df[df.honest1000.values == 0].Chat_group_all.apply(mean_word_length)
honest_len = df[df.honest1000.values != 0].Chat_group_all.apply(mean_word_length)

plt.figure(figsize=(5,4))
sns.distplot(dishonest_len, bins=10, label='dishonest')
sns.distplot(honest_len, bins=10, label='honest')
plt.xlabel('Mean Word Length')
plt.legend()
plt.tight_layout()
plt.savefig('figures/fig5b.pdf')
plt.show()

from nltk.corpus import stopwords
# nltk.download('stopwords') #uncomment if you have not yet downloaded
stop_words = set(stopwords.words('german'))

def stop_words_ratio(x):
    num_total_words = 0
    num_stop_words = 0
    for word in word_tokenize(x):
        if word in stop_words:
            num_stop_words += 1
        num_total_words += 1
    return num_stop_words / num_total_words
dishonest = df[df.honest1000.values == 0].Chat_group_all.apply(stop_words_ratio)
honest = df[df.honest1000.values != 0].Chat_group_all.apply(stop_words_ratio)

plt.figure(figsize=(5,4))
sns.distplot(dishonest, norm_hist=True, label='dishonest')
sns.distplot(honest, label='honest')

print('dishonest Mean: {:.3f}'.format(dishonest.values.mean()))
print('honest Mean: {:.3f}'.format(honest.values.mean()))
plt.xlabel('Stop Word Ratio')
plt.legend()
plt.tight_layout()
plt.savefig('figures/fig5c.pdf')
plt.show()

