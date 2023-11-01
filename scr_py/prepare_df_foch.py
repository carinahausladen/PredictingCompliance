import os
import pandas as pd

os.getcwd()

df_socio = pd.read_excel(r'data/Fochmann/Tax_Evasion_in_Groups.xls')  # socio-demographics
df_chat = pd.read_excel(r'data/Fochmann/Chats_coded.xlsx')  # text
df_chat_splchckd = pd.read_json(r'data/df_chat_spllchckd.json')  # text spellchecked

def prepare(df_chat, df_socio):
    ########
    # CHAT #
    ########
    df_chat = df_chat.sort_values(by=['Group_ID_simuliert', 'Message_Nr'])  # important for bigrams!

    # chat, subject, all text messages
    df_chat_subj = df_chat.groupby(['Group_ID_simuliert', 'Subject_ID', 'Period'])['Chat'].apply(list).to_frame()
    df_chat_subj.rename(columns={'Chat': 'Chat_subject'}, inplace=True)
    df_chat_subj['Chat_subject'] = df_chat_subj['Chat_subject'].str.join(' ')

    # chat, group, all text messages
    df_chat_group = df_chat.groupby(['Group_ID_simuliert', 'Period'])['Chat'].apply(list).to_frame()
    df_chat_group.rename(columns={'Chat': 'Chat_group_all'}, inplace=True)
    df_chat_group['Chat_group_all'] = df_chat_group['Chat_group_all'].str.join(' ')

    # chat, group, only text with label
    df_chat_w_label = df_chat.copy()
    col_list = list(df_chat_w_label)[11:df_chat_w_label.shape[1]]
    df_chat_w_label['sums'] = df_chat[col_list].sum(axis=1)
    df_chat_w_label = df_chat_w_label.loc[df_chat_w_label['sums'] > 0]  # only those chat w label!

    df_chat_w_label = df_chat_w_label.groupby(['Group_ID_simuliert', 'Period'])['Chat'].apply(list).to_frame()  # group
    df_chat_w_label.rename(columns={'Chat': 'Chat_group_label'}, inplace=True)
    df_chat_w_label['Chat_group_label'] = df_chat_w_label['Chat_group_label'].str.join(' ')

    # label (not text-messages!), group
    df_help = df_chat.iloc[:, 11:]  # select only label-columns
    labels = []
    for index, row in df_help.iterrows():
        labels.append((df_help.loc[index] == 1)[lambda x: x].index.tolist())
    df_label = df_chat.copy()
    df_label['Tags'] = labels

    df_label = df_label.groupby(['Group_ID_simuliert', 'Period'])['Tags'].apply(list).to_frame()  # group
    df_label['Tags'] = df_label['Tags'].apply(lambda x: [item for sublist in x for item in sublist])  # flat list
    df_label['Tags'] = df_label['Tags'].str.join(', ')

    # join df_chats
    df_chat_comb = df_chat_subj.join(df_chat_group)
    df_chat_comb = df_chat_comb.join(df_chat_w_label)
    df_chat_comb = df_chat_comb.join(df_label)

    #########
    # SOCIO #
    #########
    df_socio = df_socio.loc[(df_socio['treatment'] == 'G-G-G') |  # treatments where participants chat
                            (df_socio['treatment'] == 'I-G-I') & (df_socio['period'] == 4) |
                            (df_socio['treatment'] == 'I-G-I') & (df_socio['period'] == 5) |
                            (df_socio['treatment'] == 'I-G-I') & (df_socio['period'] == 6)]

    df_socio = df_socio.loc[:,
               ['group_id_simuliert', 'subject_id', 'period', 'declared_income_final', 'teilnehmer_weiblich', 'alter']]
    df_socio.rename(columns={'group_id_simuliert': 'Group_ID_simuliert',
                             'subject_id': 'Subject_ID',
                             'period': 'Period'}, inplace=True)
    df_socio = df_socio.set_index(['Group_ID_simuliert', 'Period', 'Subject_ID'])

    # add Column for agreement
    df_socio_group = df_socio.groupby(['Group_ID_simuliert', 'Period'])['declared_income_final'].apply(list)

    def checkEqual2(iterator):
        return len(set(iterator)) <= 1

    df_socio_group = df_socio_group.apply(lambda x: checkEqual2(x))  # flat list
    df_socio['equal'] = df_socio_group  # add column

    #########
    # merge #
    #########
    # merge chat and socio
    df_all = df_socio.join(df_chat_comb)

    # select single or group chat based on agree
    df_all['Chat_sel'] = df_all['Chat_group_all'].where(df_all['equal'], other=df_all['Chat_subject'])

    return df_all


df_p = prepare(df_chat, df_socio)
df_p.to_json('data/df_chat_socio.json')
df_p.to_csv('data/df_chat_socio.csv')  # only csv can preserve the index

df_p = prepare(df_chat_splchckd, df_socio)
df_p.to_json('data/df_chat_socio_splchckd.json')
df_p.to_csv('data/df_chat_socio_splchckd.csv')
