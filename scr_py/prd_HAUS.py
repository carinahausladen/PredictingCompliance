"makign out of sample predictions for df Hausladen"

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from setup import prepare_docs_haus
from utility import print_model_metrics, tfdf_embdngs
import pickle

df_chat_hours = pd.read_csv('data/df_chat_hours.csv')
df_prep_new, all_docs_new = prepare_docs_haus(df_chat_hours, y="honest10", X="Chat_subject", dv="player.hours_stated")
df_prep_new["honest30"].value_counts()

# Embeddings
with open('data/fitted_tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
X = tfidf_vectorizer.transform(all_docs_new.new_docs)

stacking_clf = pickle.load(open("data/best_model.sav", 'rb'))

y = df_prep_new["honest10"]
y_pred_prob = stacking_clf.predict_proba(X.toarray())
metrics_10 = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)

y = df_prep_new["honest30"]
y_pred_prob = stacking_clf.predict_proba(X.toarray())
metrics_30 = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)

y = df_prep_new["honestmean"]
y_pred_prob = stacking_clf.predict_proba(X.toarray())
metrics_mean = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)

# LATEX
metrics_df = pd.DataFrame([metrics_mean, metrics_30, metrics_10])
metrics_df.index = ['> mean', '> 30', '> 10']
metrics_df = (metrics_df * 100).round(decimals=3)
latex_output = metrics_df.to_latex(header=["f1score", "precision", "recall", "AUC", "accuracy"],
                                   float_format="{:0.1f}".format)
print(latex_output)