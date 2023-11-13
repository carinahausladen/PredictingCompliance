import os
import pandas as pd

os.getcwd()

df_socio = pd.read_excel(r'data/Fochmann/Tax_Evasion_in_Groups.xls')  # socio-demographics
df_chat = pd.read_excel(r'data/Fochmann/Chats_coded.xlsx')  # text
df_chat_splchckd = pd.read_json(r'data/df_chat_spllchckd.json')  # text spellchecked

columns_to_sum = [
 'Money - pro honesty', 'Money - pro lying', 'Steuern allgemein',
 'argument für Steuern (pro honesty)', 'argument gegen Steuern (pro lying)',
 'Risk - allgemein', 'Risk - pro honesty', 'Risk - pro lying',
 'Honesty - allgemein', 'Honesty - pro honesty', 'Honesty - pro lying',
 'Zahlenvorschlag - allgemein', 'Zahlenvorschlag pro honesty',
 'Zahlenvorschlag pro lying', 'Previous Strategy - keep  - pro honesty',
 'Previous Strategy - keep - pro lying', 'Previous Strategy - change - pro honesty',
 'Previous Strategy - change - pro lying', 'Realität/Spiel',
 'Steuergerechtigkeit', 'Steuerehrlichkeit', 'argument insecurity - allgemein',
 'argument insecurity honest', 'argument insecurity lie', 'argument rules - allgemein',
 'argument rules yes', 'argument rules no', 'argument others - allgemein',
 'argument others honest', 'argument others lie', 'argument consequences - allgemein',
 'consequences positive', 'consequences negative', 'taxes_new']


#df_chat = df_chat_splchckd
def prepare(df_chat, df_socio):
    ########
    # CHAT #
    ########
    df_chat = df_chat.sort_values(by=['Group_ID_simuliert', 'Message_Nr'])  # important for bigrams!

    df_chat['Chat'] = df_chat['Chat'].fillna('')
    df_chat['Chat'] = df_chat['Chat'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))

    # chat, subject, all text messages
    df_chat_subj = df_chat.groupby(['Group_ID_simuliert', 'Subject_ID', 'Period'])['Chat'].apply(list).to_frame()
    df_chat_subj.rename(columns={'Chat': 'Chat_subject'}, inplace=True)
    df_chat_subj['Chat_subject'] = df_chat_subj['Chat_subject'].str.join(' ')
    df_chat_subj['Chat_subject'] = df_chat_subj['Chat_subject'].fillna('kein chat') #in some rounds there was no chat
    df_chat_subj['Chat_subject'] = df_chat_subj['Chat_subject'].apply(lambda x: x if x != "" else "kein chat")

    # chat, group, all text messages
    df_chat_group = df_chat.groupby(['Group_ID_simuliert', 'Period'])['Chat'].apply(list).to_frame()
    df_chat_group.rename(columns={'Chat': 'Chat_group_all'}, inplace=True)
    df_chat_group['Chat_group_all'] = df_chat_group['Chat_group_all'].str.join(' ')
    df_chat_group['Chat_group_all'] = df_chat_group['Chat_group_all'].fillna('kein chat')
    df_chat_group['Chat_group_all'] = df_chat_group['Chat_group_all'].apply(lambda x: x if x != "" else "kein chat")

    # chat, group, only text with label
    df_chat_w_label = df_chat.copy()
    df_chat_w_label['sums'] = df_chat_w_label[columns_to_sum].sum(axis=1)
    df_chat_w_label = df_chat_w_label.loc[df_chat_w_label['sums'] > 0]  # only those chat w label!
    df_chat_w_label = df_chat_w_label.groupby(['Group_ID_simuliert', 'Period'])['Chat'].apply(list).to_frame()  # group
    df_chat_w_label.rename(columns={'Chat': 'Chat_group_label'}, inplace=True)
    df_chat_w_label['Chat_group_label'] = df_chat_w_label['Chat_group_label'].str.join(' ')
    df_chat_w_label['Chat_group_label'] = df_chat_w_label['Chat_group_label'].fillna('kein chat')
    df_chat_w_label['Chat_group_label'] = df_chat_w_label['Chat_group_label'].apply(lambda x: x if x != "" else "kein chat")

    # join df_chats
    df_chat_comb = df_chat_subj.join(df_chat_group)
    df_chat_comb = df_chat_comb.join(df_chat_w_label)
    df_chat_comb['Chat_group_label'] = df_chat_comb['Chat_group_label'].fillna('kein chat')

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
    df_all['Chat_subject'] = df_all['Chat_subject'].fillna('kein chat') #there are rounds with no chat
    df_all['Chat_group_all'] = df_all['Chat_group_all'].fillna('kein chat')
    df_all['Chat_group_label'] = df_all['Chat_group_label'].fillna('kein chat')

    # select single or group chat based on agree
    df_all['Chat_sel'] = df_all['Chat_group_all'].where(df_all['equal'], other=df_all['Chat_subject'])

    return df_all


df_p = prepare(df_chat, df_socio)
df_p.to_json('data/df_chat_socio.json')
df_p.to_csv('data/df_chat_socio.csv')  # only csv can preserve the index

df_p = prepare(df_chat_splchckd, df_socio)
df_p.to_json('data/df_chat_socio_splchckd.json')
df_p.to_csv('data/df_chat_socio_splchckd.csv')


