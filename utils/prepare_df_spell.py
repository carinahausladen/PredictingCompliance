import multiprocessing as mp

import pandas as pd
from spellchecker import SpellChecker
from tqdm import tqdm

'note that parallel version only works in Ipython console'

spell = SpellChecker(language='de')
df_chat = pd.read_csv("data/chat_hours.csv")


def spellcheck_chat(df):
    chat_cols = ['Chat_subject', 'Chat_group_all', 'Chat_sel']
    df.loc[:, chat_cols] = df.loc[:, chat_cols].fillna('kein_Chat').astype(str)
    df.loc[:, chat_cols] = df.loc[:, chat_cols].applymap(lambda x: x if x != "" else "kein_Chat")

    spell = SpellChecker(language='de') # assuming you are using German language. If not, replace 'de' with appropriate language.

    for chat_col in chat_cols:
        chat_group_str = ' '.join(map(str, df[chat_col]))  # making one string out of all chat messages per group
        words = spell.split_words(chat_group_str)
        words_deduplicate = list(set(words))
        su = spell.unknown(words_deduplicate)

        # parallel processing
        pool = mp.Pool(10)
        result = pool.map(spell.correction, tqdm(su))
        pool.close()

        result_zip = dict(zip(su, result))
        result_zip = {key: val for key, val in result_zip.items() if key != val}

        df[chat_col] = df[chat_col].apply(lambda x: spell.split_words(x))
        df[chat_col] = df[chat_col].apply(lambda x: [result_zip[word] if word in result_zip else word for word in x])
        df[chat_col] = df[chat_col].apply(lambda x: ', '.join(i for i in x if i is not None))

    return df


df_checked = spellcheck_chat(df_chat)
df_checked.to_csv('data/chat_hours_spll.csv')

# len(set(spell.split_words(chat_group_str)))  # length of unique words
