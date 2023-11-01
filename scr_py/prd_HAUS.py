############## make predictions on Hausladen
import pandas as pd
from pymagnitude import *
from sklearn.feature_extraction.text import TfidfVectorizer

from setup import prepare_docs_haus
from utility import print_model_metrics, tfdf_embdngs

df_chat_hours = pd.read_csv('/Users/carinahausladen/FHM/data/Hausladen/df_chat_hours.csv')
df_prep_new, all_docs_new = prepare_docs_haus(df_chat_hours, y="honest10", X="Chat_subject", dv="player.hours_stated")

df_prep_new["honest30"].value_counts()

# Embeddings
tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=lambda x: x)
w2v = Magnitude('data/w2v.magnitude')
tfidf.fit(all_docs_new.new_docs)
idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
X_tfidf_w2v_pre = tfdf_embdngs(all_docs_new.new_docs, w2v, dict_tf=idf_dict)

X = X_tfidf_w2v_pre
y = df_prep_new["honest10"]

# load best model & predict
import pickle
from cmpr_clf import StackingClassifier
stacking_clf = pickle.load(open("data/best_model.sav", 'rb'))
y_pred_prob = stacking_clf.predict_proba(X)

metrics_stack_new_mean = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)
metrics_stack_new_10 = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)
metrics_stack_new_30 = print_model_metrics(y, y_pred_prob, confusion=False, return_metrics=True)
