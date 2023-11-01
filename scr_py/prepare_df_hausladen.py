import pandas as pd

df_hours = pd.read_csv('data/hours_stated.csv') # created by combine.R
df_chat = pd.read_csv('data/chat.csv')


def prepare(df_chat, df_hours):
    ########
    # CHAT #
    ########
    df_chat = df_chat.sort_values(by=['channel', 'timestamp'])  # important for bigrams!

    # chat, subject, all text messages
    df_chat_subj = df_chat.groupby(['channel', 'participant__code'])['body'].apply(list).to_frame()
    df_chat_subj.rename(columns={'body': 'Chat_subject'}, inplace=True)
    df_chat_subj['Chat_subject'] = df_chat_subj['Chat_subject'].str.join(' ')

    # chat, group, all text messages
    df_chat_group = df_chat.groupby(['channel'])['body'].apply(list).to_frame()
    df_chat_group.rename(columns={'body': 'Chat_group_all'}, inplace=True)
    df_chat_group['Chat_group_all'] = df_chat_group['Chat_group_all'].str.join(' ')

    # join df_chats
    df_chat_comb = df_chat_subj.join(df_chat_group)

    #########
    # SOCIO #
    #########
    df_hours = df_hours.set_index(['session.code', 'group.id_in_subsession'])

    # add Column for agreement
    df_hours_temp = df_hours.groupby(['session.code', 'group.id_in_subsession'])['player.hours_stated'].apply(list)

    def checkEqual2(iterator):
        return len(set(iterator)) <= 1

    df_hours_temp = df_hours_temp.apply(lambda x: checkEqual2(x))  # flat list
    df_hours['equal'] = df_hours_temp  # add column

    #########
    # merge #
    #########

    # merge chat and hours stated
    df_hours = df_hours.reset_index()
    df_hours = df_hours.set_index(['participant.code'])

    df_chat_comb = df_chat_comb.reset_index()
    df_chat_comb = df_chat_comb.set_index(['participant__code'])

    df_all = df_hours.join(df_chat_comb)

    # select single or group chat based on agree
    df_all['Chat_sel'] = df_all['Chat_group_all'].where(df_all['equal'], other=df_all['Chat_subject'])

    return df_all


df_p = prepare(df_chat, df_hours)
df_p.to_csv('data/df_chat_hours.csv')  # only csv can preserve the index
