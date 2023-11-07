import sys
import os
import pandas as pd
import spacy

sys.path.append(os.path.abspath("scr_py"))
from UtilWordEmbedding import DocPreprocess

def prepare_X_y_haus(df, dv):
    # prepare x vars
    chat_cols = ['Chat_subject', 'Chat_group_all', 'Chat_sel']
    df.loc[:, chat_cols] = df.loc[:, chat_cols].fillna('kein_chat').astype(str)
    df[chat_cols] = df[chat_cols].apply(lambda col: col.map(lambda x: x if x != "" else "kein_chat"))

    # generate y vars; dv=declared_income
    df['honest10'] = (df[dv] <= 10).astype(int)  # Compliance: 10:1, 11-60:0
    df['honest30'] = (df[dv] < 30).astype(int)  # Compliance: 10-29:1, 30-60:0
    df['honestmean'] = (df[dv] < df[dv].mean()).astype(int)  # Compliance: 10-mean:1, mean-60:0

    return df

def prepare_docs_haus(df, y, X, dv):
    df = prepare_X_y_haus(df, dv)

    pattern = '|'.join(["XD", "xd", "xD",
                        "X-D", "x-d", "x-D",
                        ":D", ";D",
                        ":-D", ";-D",
                        ":\)", ";\)",
                        ":-\)", ";-\)", "haha"
                        ])
    df.loc[:, X] = df.loc[:, X].str.replace(pattern, " smiley ", regex=True)

    nlp = spacy.load("de_core_news_sm")  # .venv/bin/python -m spacy download de
    stop_words = spacy.lang.de.stop_words.STOP_WORDS
    all_docs = DocPreprocess(nlp, stop_words, df[X], df[y])

    return df, all_docs


def prepare_X_y(df, dv):

    # generate y vars; dv=declared_income
    df['honest1000'] = (df[dv] >= 1000).astype(int)
    df['honest500'] = (df[dv] > 500).astype(int)  # MODE 0-500:0, 501-1000:1 (minority)
    df['honestmean'] = (df[dv] > df[dv].mean()).astype(int) # MEAN

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
    df.loc[:, X] = df.loc[:, X].str.replace(pattern, " smiley ", regex=True)

    nlp = spacy.load('de_core_news_sm')
    #nlp = spacy.load('de')  # .venv/bin/python -m spacy download de
    stop_words = spacy.lang.de.stop_words.STOP_WORDS

    #missing_rows = df[df[X].isna()]
    #print(missing_rows[X])

    all_docs = DocPreprocess(nlp, stop_words, df[X], df[y])

    return df, all_docs

#testing
#df = pd.read_json('data/df_chat_socio.json')
#dv="declared_income_final"
#y="honest1000"
#X="Chat_subject"