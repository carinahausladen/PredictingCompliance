import pandas as pd
import language_tool_python
from tqdm import tqdm

df_chat = pd.read_excel(r'data/Fochmann/Chats_coded.xlsx')  # includes text
tqdm.pandas()
tool = language_tool_python.LanguageTool('de-DE')

def correct_text(text):
    # Check if the text is not a string (which includes checking for NaN values in pandas)
    if not isinstance(text, str):
        return text  # or return "" if you want to replace NaNs with empty strings
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

df_chat['Corrected_Chat'] = df_chat['Chat'].progress_apply(correct_text)

df_chat.drop('Chat', axis=1, inplace=True)
df_chat.rename(columns={'Corrected_Chat': 'Chat'}, inplace=True)

df_chat.to_json('data/df_chat_spllchckd.json')
df_chat.to_csv('data/df_chat_spllchckd.csv')