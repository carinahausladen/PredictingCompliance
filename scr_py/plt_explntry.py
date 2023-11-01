"""
script plots descriptive statistics for y and X
I should not draw upon the original data! I need the prepared data which only includes the selected groups!
Includes data from old and new experiment.
Is not very clean! at some point data-sets need to be switched!
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import word_tokenize

from setup import prepare_docs, prepare_docs_haus

os.chdir("/Users/carinahausladen/FHM/analysis/")

df = pd.read_json('data/df_chat_socio.json')
df_prep, all_docs = prepare_docs(df, y="honest1000", X="Chat_subject", dv="declared_income_final")

df_chat_hours = pd.read_csv('/Users/carinahausladen/FHM/data/Hausladen/df_chat_hours.csv')
df_prep_new, all_docs_new = prepare_docs_haus(df_chat_hours, y="honest10", X="Chat_subject", dv="player.hours_stated")

#########
# plot y#
#########
'why do I plot the original data?? I need to plot the prepared data!!'
# df2 = pd.read_excel(r'data/Tax_Evasion_in_Groups.xls', sheet_name="Sheet1")
df['declared_income_final'].hist(bins=20, grid=False)
plt.title("")
plt.xlabel('declared income', fontsize=18)
plt.ylabel('frequency', fontsize=16)
#plt.savefig('figures/income.eps', format="eps")
plt.savefig('figures/income.pdf')
plt.show()

# add stacked barcharts
plot_df = pd.DataFrame()
# plot_df["3label"] = df_prep["honest3label"].value_counts()
plot_df["1000"] = df_prep["honest1000"].value_counts()
plot_df["500"] = df_prep["honest500"].value_counts()
plot_df["mean"] = df_prep["honestmean"].value_counts()
plot_df = plot_df.rename(index={1: "honest",
                                2: "undefined",
                                0: "dishonest"})

plot_df.T.plot(kind="bar", stacked=True)
plt.legend(loc='best')  # , bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.xlabel('threshold to binarize reported income', fontsize=18)
plt.ylabel('frequency', fontsize=16)
plt.savefig('figures/label_dist.eps', format='eps')
#plt.savefig('figures/label_dist.pdf')
plt.show()

# HAUSLADEN DF
#df_new = pd.read_csv('../data/Hausladen/hours_stated.csv')
df_new = pd.read_csv('df_chat_hours.csv')

df_new[['player.hours_stated']].hist(bins=20, grid=False)
plt.title("")
plt.xlabel('hours stated', fontsize=18)
plt.ylabel('frequency', fontsize=16)
#plt.savefig('figures/exp_new_income.eps', format="eps")
plt.savefig('figures/exp_new_income.pdf', transparent=True)
plt.show()

# add stacked barcharts
plot_df = pd.DataFrame()
# plot_df["3label"] = df_prep_new["honest3label"].value_counts()
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
#plt.savefig('figures/label_dist_new.eps', format='eps')
plt.savefig('figures/label_dist_new.pdf', transparent=True)
plt.show()

#########
# plot X#
#########
all_docs = all_docs_new  # just for the new experiment
df = df_chat_hours
df = df.rename(columns={"honest10": "honest1000"})

print(df.Chat_group_all.str.len())
s = df.Chat_group_all.str.len().sort_values().index
print(df.Chat_group_all.reindex(s))

# distribution of number of words
dishonest = [len(word_tokenize(Chat_group_all)) for Chat_group_all in
             df[df.honest1000.values == 0].Chat_group_all.values]
honest = [len(word_tokenize(Chat_group_all)) for Chat_group_all in df[df.honest1000.values != 0].Chat_group_all.values]

plt.figure(figsize=(5,4))
sns.distplot(dishonest, bins=10, label='dishonest')
sns.distplot(honest, bins=10, label='honest')
plt.legend()
# plt.title('Distribution of Number of Words')
plt.xlabel('Number of Words')
plt.tight_layout()
plt.savefig("figures/yana_nmb_new.pdf")
#plt.savefig("figures/yana_nmb.pdf")
plt.show()


# mean word length?
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
# plt.title('Distribution of Mean Word Length')
plt.xlabel('Mean Word Length')
plt.legend()
plt.tight_layout()
plt.savefig("figures/yana_lnght_new.pdf")
#plt.savefig("figures/yana_lnght.pdf")
plt.show()

# stopwords
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')
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
# plt.title('Distribution of Stop-word Ratio')
plt.xlabel('Stop Word Ratio')
plt.legend()
plt.tight_layout()
plt.savefig('figures/yana_stp_new.pdf')
#plt.savefig('figures/yana_stp.pdf')
plt.show()

### find overlapping words ###
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_tfidf = vectorizer.fit_transform(all_docs.new_docs)
voc_foch = vectorizer.vocabulary_

vectorizer = CountVectorizer()
X_tfidf = vectorizer.fit_transform(all_docs_new.new_docs)
voc_haus = vectorizer.vocabulary_

len(voc_foch)
len(voc_haus)

keys_a = set(voc_foch.keys())
keys_b = set(voc_haus.keys())
intersection = keys_a & keys_b
len(intersection)
