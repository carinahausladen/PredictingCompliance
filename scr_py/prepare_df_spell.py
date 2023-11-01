import multiprocessing as mp
import os

import pandas as pd
from spellchecker import SpellChecker
from tqdm import tqdm

'note that parallel version only works in Ipython console'
spell = SpellChecker(language='de')

df_chat = pd.read_excel(r'data/Fochmann/Chats_coded.xlsx')  # includes text


df_chat['Chat']
df = df_chat #for testing
def spellcheck_chat(df):
    chat_cols = ['Chat']
    df.loc[:, chat_cols] = df.loc[:, chat_cols].fillna('kein_Chat').astype(str)
    df[chat_cols] = df[chat_cols].apply(lambda col: col.map(lambda x: x if x != "" else "kein_Chat"))

    chat_group_str = ' '.join(map(str, df_chat['Chat']))  # making one string out of all chat messages per group
    words = spell.split_words(chat_group_str)
    words_deduplicate = list(set(words))
    su = spell.unknown(words_deduplicate)
    spell_crr_dict = dict()

    # parallel processing
    pool = mp.Pool(10)
    result = pool.map(spell.correction, tqdm(su))
    pool.close()

    result_zip = dict(zip(su, result))
    result_zip = {key: val for key, val in result_zip.items() if key != val}

    df['Chat'] = df['Chat'].apply(lambda x: spell.split_words(x))
    df['Chat'] = df['Chat'].apply(lambda x: [result_zip[word] if word in result_zip else word for word in x])
    df['Chat'] = df['Chat'].apply(lambda x: " ".join(x))

    return df


df_checked = spellcheck_chat(df_chat)
df_checked.to_json('data/df_chat_spllchckd.json')
df_checked.to_csv('data/df_chat_spllchckd.csv')

# len(set(spell.split_words(chat_group_str)))  # length of unique words
