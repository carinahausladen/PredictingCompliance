import pandas as pd
import language_tool_python
from tqdm import tqdm

df_chat = pd.read_excel(r'data/Fochmann/Chats_coded.xlsx')  # includes text
tqdm.pandas()
tool = language_tool_python.LanguageTool('de-DE')

def correct_text(text):
    if not isinstance(text, str):
        return text, 0  # return the original text (or NaN) and zero corrections
    matches = tool.check(text)
    number_of_corrections = len(matches)  # count the matches as the number of corrections
    corrected_text = language_tool_python.utils.correct(text, matches)
    # count tokens in the original text
    tokens_in_text = len(text.split())
    return corrected_text, number_of_corrections, tokens_in_text

# Apply the correction and count corrections and tokens
df_chat[['Corrected_Chat', 'Number_of_Corrections', 'Tokens_in_Text']] = df_chat['Chat'].progress_apply(
    lambda x: pd.Series(correct_text(x))
)

# Calculate the total number of corrections and tokens
total_corrections = df_chat['Number_of_Corrections'].sum()
total_tokens = df_chat['Tokens_in_Text'].sum()

# Calculate the percentage of tokens corrected
percentage_of_tokens_corrected = (total_corrections / total_tokens) * 100 if total_tokens else 0

print(f"Total number of corrections made: {total_corrections}")
print(f"Total number of tokens: {total_tokens}")
print(f"Percentage of tokens corrected: {percentage_of_tokens_corrected:.2f}%")


df_chat.drop('Chat', axis=1, inplace=True)
df_chat.drop('Number_of_Corrections', axis=1, inplace=True)
df_chat.drop('Tokens_in_Text', axis=1, inplace=True)
df_chat.rename(columns={'Corrected_Chat': 'Chat'}, inplace=True)

df_chat.to_json('data/df_chat_spllchckd.json')
df_chat.to_csv('data/df_chat_spllchckd.csv')