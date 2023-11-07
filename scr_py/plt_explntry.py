"""
script plots descriptive statistics for y and X
"""
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath("scr_py"))
from setup import prepare_docs, prepare_docs_haus

# Fochmann
df_fochmann = pd.read_json('../data/df_chat_socio.json')
df_prep_foch, all_docs_haus = prepare_docs(df_fochmann, y="honest1000", X="Chat_subject", dv="declared_income_final")

# Hausladen
df_hausladen = pd.read_csv('../data/df_chat_hours.csv')
df_prep_haus, all_docs_haus = prepare_docs_haus(df_hausladen, y="honest10", X="Chat_subject", dv="player.hours_stated")

#########
# plot FOCHMANN#
#########
mean_income = df_fochmann['declared_income_final'].mean()
fig, axs = plt.subplots(figsize=(4, 2), ncols=2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.7})

# Histogram
sns.histplot(df_fochmann['declared_income_final'], bins=20, ax=axs[0], color='#FFC107', edgecolor="none")
axs[0].axvline(998, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(500, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(mean_income, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(0, color='#D41159', linestyle='-', linewidth=1.5, label='Full Non-Compliance')
axs[0].axvline(1001, color='#1A85FF', linestyle='-', linewidth=1.5, label='Full Compliance')
axs[0].set_xlabel('Declared Income')
axs[0].set_ylabel('Frequency')
legend_elements = [
    Line2D([0], [0], color='grey', linestyle='--', label='Thresholds'),
    Line2D([0], [0], color='#D41159', linestyle='-', label='Full Non-Compliance'),
    Line2D([0], [0], color='#1A85FF', linestyle='-', label='Full Compliance')
]
axs[0].legend(handles=legend_elements, loc='upper right', prop={'size': 8})
axs[0].set_xticks([0, mean_income, 500, 1000])
axs[0].set_xticklabels([0, f"{mean_income:.0f}", "500", "1000"])

# Stacked barcharts
columns = ["honestmean", "honest500", "honest1000"]
plot_df = pd.DataFrame({col: df_prep_foch[col].value_counts() for col in columns})
rename_dict = {1: "honest", 2: "undefined", 0: "dishonest"}
plot_df = plot_df.rename(index=rename_dict)
plot_df.T.plot(kind="bar", stacked=True, ax=axs[1], color=['#D41159', '#1A85FF', 'grey'])
axs[1].set_xticklabels(['mean', 'median', 'max'], rotation=45, fontsize=8)  # Adjust rotation for better readability
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Frequency')
axs[1].set_ylim(0, 800)
axs[1].get_legend().remove()
fig.suptitle('Experiment I', fontsize=14)

fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('../figures/y_fochmann.pdf', bbox_inches='tight')
plt.show()


#########
#PLOT HAUSLADEN
#########
mean_income = round(df_hausladen['player.hours_stated'].mean())
fig, axs = plt.subplots(figsize=(4, 2), ncols=2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.7})

# Histogram
sns.histplot(df_hausladen['player.hours_stated'], bins=20, ax=axs[0], color='#FFC107', edgecolor="none")
axs[0].axvline(10.5, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(30, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(mean_income, color='grey', linestyle='--', linewidth=2)
axs[0].axvline(10, color='#D41159', linestyle='-', linewidth=1.5, label='Full Compliance')
axs[0].axvline(60, color='#1A85FF', linestyle='-', linewidth=1.5, label='Full Non-Compliance')
axs[0].set_xlabel('Hours Stated')
axs[0].set_ylabel('Frequency')
legend_elements = [
    Line2D([0], [0], color='grey', linestyle='--', label='Thresholds'),
    Line2D([0], [0], color='#D41159', linestyle='-', label='Full Non-Compliance'),
    Line2D([0], [0], color='#1A85FF', linestyle='-', label='Full Compliance')
]
axs[0].legend(handles=legend_elements, loc='upper right', prop={'size': 8})
axs[0].set_xticks([10, mean_income, 30, 60])
axs[0].set_xticklabels([10, f"{mean_income:.0f}", "30", "60"], fontsize=8)

# Stacked barcharts
ordered_columns = ["honest10", "honest30", "honestmean"]
plot_df = pd.DataFrame({col: df_prep_haus[col].value_counts() for col in ordered_columns})
rename_dict = {1: "honest", 2: "undefined", 0: "dishonest"}
plot_df = plot_df.rename(index=rename_dict)
plot_df.T.plot(kind="bar", stacked=True, ax=axs[1], color=['#D41159', '#1A85FF', 'grey'])
axs[1].set_xticklabels(['min', 'median', 'mean'], rotation=45, fontsize=8)
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Frequency')
axs[1].get_legend().remove()
fig.suptitle('Experiment II', fontsize=14)

plt.tight_layout()
plt.savefig('../figures/y_hausladen.pdf', bbox_inches='tight')
plt.show()

