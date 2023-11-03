import os
import fasttext
import pandas as pd
from setup import prepare_docs
from strt_grp_sffl_splt import str_grp_splt
from imblearn.over_sampling import RandomOverSampler
from utility import print_model_metrics
import numpy as np

df = pd.read_csv('data/df_chat_socio.csv')
df, df_doc = prepare_docs(df, y="honestmean", X="Chat_subject", dv="declared_income_final")
df_doc.new_docs = [x if len(x) != 0 else "kein_chat" for x in df_doc.new_docs]
ros = RandomOverSampler(random_state=42, sampling_strategy='minority')  # oversampling!

train_idx, test_idx = str_grp_splt(df,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honestmean",
                                   train_share=0.8)

# prepare y
df["honestmean"] = df["honestmean"].replace(1, '__label__honest')
df["honestmean"] = df["honestmean"].replace(0, '__label__dishonest')

df_train = df.loc[train_idx, ["honestmean", "Chat_subject"]]
df_test = df.loc[test_idx, ["honestmean", "Chat_subject"]]

# oversampling!
df_train["honestmean"].value_counts()
max_size = df_train['honestmean'].value_counts().max()
lst = [df_train]
for class_index, group in df_train.groupby('honestmean'):
    lst.append(group.sample(max_size-len(group), replace=True))
df_train = pd.concat(lst)

df_train.to_csv(r'data/chat_train.txt', header=None, index=None, sep=' ', mode='w', quoting=None)
df_test.to_csv(r'data/chat_test.txt', header=None, index=None, sep=' ', mode='w', quoting=None)
df.to_csv(r'data/chat.txt', header=None, index=None, sep=' ', mode='w', quoting=None)

# preprocess X
os.system("cat /Users/carinah/Documents/GitHub/PredictingCompliance/data/chat_train.txt | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > /Users/carinah/Documents/GitHub/PredictingCompliance/data/chat.pre.train.txt")
os.system("cat /Users/carinah/Documents/GitHub/PredictingCompliance/data/chat_test.txt | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > /Users/carinah/Documents/GitHub/PredictingCompliance/data/chat.pre.test.txt")
os.system("cat /Users/carinah/Documents/GitHub/PredictingCompliance/data/chat.txt | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > /Users/carinah/Documents/GitHub/PredictingCompliance/fastText/chat.pre.txt")

# model and training
#model = fasttext.train_supervised(input='data/chat.pre.train.txt')
model = fasttext.train_supervised(input='data/chat.pre.train.txt', autotuneValidationFile='data/chat.pre.test.txt')  # takes 5 minutes

def run_pred():
    label_l = []
    predproba_l = []

    for ti in test_idx:
        label, predproba = model.predict(df_doc.new_docs[ti], k=1)
        if label[0] == '__label__dishonest':
            label = 0
        else:
            label = 1

        if label == 1:
            predproba = predproba[0]
        else:
            predproba = 1 - predproba[0]

        label_l.append(label)
        predproba_l.append(predproba)

    return predproba_l


# define test and train
test_X = df["Chat_subject"][test_idx]
test_y = df["honestmean"][test_idx]

test_y = test_y.replace('__label__honest', 1)  # rename back to numeric
test_y = test_y.replace('__label__dishonest', 0)

# make predictions
metrics = list()
for _ in range(10):
    y_test_prob = run_pred()
    metrics.append(print_model_metrics(test_y, y_test_prob, confusion=False, verbose=False, return_metrics=True))
metrics_matrix = np.stack(metrics)  # 10x5 matrix
metrics = np.mean(metrics_matrix, axis=0)

m_ft_own = metrics
np.save("data/m_ft_own.npy", m_ft_own)

# check performance
#result_train = model.test('data/chat.pre.train.txt')
#result_test = model.test('data/chat.pre.test.txt')


# check quality of vectors
#model.words
#model.get_nearest_neighbors(':D')
#model.get_analogies("risiko", "hinterziehen", "steuer")  # this command takes a triplet


# load pre-trained vectors
#import fasttext.util
#fasttext.util.download_model('de', if_exists='ignore')  # german
#model_pre = fasttext.load_model('cc.de.300.bin')

# Prepare one for the visualzation of embeddings