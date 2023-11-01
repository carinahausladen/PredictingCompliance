import spacy

from UtilWordEmbedding import DocPreprocess


def prepare_X_y_haus(df, dv):
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


def prepare_docs_haus(df, y, X, dv):
    df = prepare_X_y_haus(df, dv)

    pattern = '|'.join(["XD", "xd", "xD",
                        "X-D", "x-d", "x-D",
                        ":D", ";D",
                        ":-D", ";-D",
                        ":\)", ";\)",
                        ":-\)", ";-\)", "haha"
                        ])
    df.loc[:, X] = df.loc[:, X].str.replace(pattern, "smiley", regex=True)

    nlp = spacy.load('de')  # .venv/bin/python -m spacy download de
    stop_words = spacy.lang.de.stop_words.STOP_WORDS
    all_docs = DocPreprocess(nlp, stop_words, df[X], df[y])

    return df, all_docs


def prepare_X_y(df, dv):
    # prepare x vars
    chat_cols = ['Chat_subject', 'Chat_group_all', 'Chat_sel', 'Chat_group_label', 'Tags']
    df.loc[:, chat_cols] = df.loc[:, chat_cols].fillna('kein_chat').astype(str)
    df.loc[:, chat_cols] = df.loc[:, chat_cols].applymap(lambda x: x if x != "" else "kein_chat")

    # generate y vars; dv=declared_income
    df['honest1000'] = (df[dv] >= 1000).astype(int)
    df['honest500'] = (df[dv] > 500).astype(int)  # 0-499:0, 500-1000:1 (minority)
    df['honestmean'] = (df[dv] > df[dv].mean()).astype(int)

    def define_classes(x):
        if x == 1000:
            return 1
        elif x == 0:
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

    nlp = spacy.load('de')  # .venv/bin/python -m spacy download de
    stop_words = spacy.lang.de.stop_words.STOP_WORDS
    all_docs = DocPreprocess(nlp, stop_words, df[X], df[y])

    return df, all_docs