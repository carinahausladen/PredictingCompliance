# https://github.com/anirudhshenoy/text-classification-small-datasets/blob/master/notebooks/models_ensembles_tuning.ipynb
'vectorizer chosen: tfidf (3rd best in my plot, but simplest!)'

import os
import sys
sys.path.append(os.path.abspath("scr_py"))

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer

from setup import prepare_docs
from strt_grp_sffl_splt import str_grp_splt
from utility import print_model_metrics, run_grid_search, fit_n_times, tfdf_embdngs
import pickle

### setup I
df = pd.read_csv('../data/df_chat_socio.csv')  # 855 rows, 9 columns
df_new, all_docs = prepare_docs(df, X="Chat_subject", y="honestmean", dv="declared_income_final")
all_docs.new_docs = [x if len(x) != 0 else "kein chat" for x in all_docs.new_docs]
ros = RandomOverSampler(random_state=42, sampling_strategy='minority')  # oversampling!
df["honestmean"].value_counts()  # 1: minority = compliance

# Embeddings
def my_preprocessor(x):
    return x
tfidf = TfidfVectorizer(input='content', lowercase=False, preprocessor=my_preprocessor)
X_tfidf = tfidf.fit_transform(all_docs.new_docs)
with open('../data/fitted_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)


# split
train_idx, test_idx = str_grp_splt(df_new,
                                   grp_col_name="Group_ID_simuliert",
                                   y_col_name="honest500",
                                   train_share=0.8)
train_X = X_tfidf[train_idx]
test_X = X_tfidf[test_idx]
train_y = df_new["honestmean"][train_idx]
test_y = df_new["honestmean"][test_idx]
train_X, train_y = ros.fit_resample(train_X, train_y)

df_new["honestmean"].value_counts()  # minority label is 1!
train_y.value_counts()
test_y.value_counts()
train_X, train_y = ros.fit_resample(train_X, train_y)
train_y.value_counts()

### Logistic Regression
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log_loss', random_state=42)
lr_params = {'alpha': [10 ** (-x) for x in range(7)],
             'penalty': ['l1', 'l2', 'elasticnet'],
             'l1_ratio': [0.15, 0.25, 0.5, 0.75]}
best_params, best_f1 = run_grid_search(lr, lr_params, train_X, train_y)

print('Best Parameters : {}'.format(best_params))
print('Best F1 : {}'.format(best_f1))

lr = SGDClassifier(loss='log_loss',
                   alpha=best_params['alpha'],
                   penalty=best_params['penalty'],
                   l1_ratio=best_params['l1_ratio'],
                   random_state=42)
metrics_lr = fit_n_times(lr, train_X, train_y, test_X, test_y)

### SVM
from sklearn.svm import SVC

svm = SVC(probability=True, random_state=42)
svm_params = {'C': [10 ** (x) for x in range(-1, 4)],
              'kernel': ['poly', 'rbf', 'linear'],
              'degree': [2, 3]}

best_params, best_f1 = run_grid_search(svm, svm_params, train_X, train_y)

print('Best Parameters : {}'.format(best_params))
print('Best F1 : {}'.format(best_f1))

svm = SVC(C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'], probability=True, random_state=42)
metrics_svm = fit_n_times(svm, train_X, train_y, test_X, test_y)

### KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_jobs=-1)
knn_params = {'n_neighbors': [3, 5, 7, 9, 15, 31],
              'weights': ['uniform', 'distance']
              }

best_params, best_f1 = run_grid_search(knn, knn_params, train_X, train_y)
print('Best Parameters : {}'.format(best_params))

knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'], n_jobs=-1)

metrics_knn = fit_n_times(knn, train_X, train_y, test_X, test_y)

### Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

rf_params = {'n_estimators': [10, 100, 250, 500, 1000],
             'max_depth': [None, 3, 7, 15],
             'min_samples_split': [2, 5, 15]
             }

best_params, best_f1 = run_grid_search(rf, rf_params, train_X, train_y)

print('Best Parameters : {}'.format(best_params))
rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                            min_samples_split=best_params['min_samples_split'],
                            max_depth=best_params['max_depth'],
                            n_jobs=-1,
                            random_state=42)
metrics_rf = fit_n_times(rf, train_X, train_y, test_X, test_y)

### XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(n_jobs=-1, random_state=42)

xgb_params = {'n_estimators': [10, 100, 200, 500],
              'max_depth': [1, 2, 3, 7],
              'learning_rate': [0.1, 0.2, 0.01, 0.3],
              'reg_alpha': [0, 0.1, 0.2]
              }

best_params, best_f1 = run_grid_search(xgb, xgb_params, train_X, train_y)

print('Best Parameters : {}'.format(best_params))
xgb = XGBClassifier(n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate'],
                    max_depth=best_params['max_depth'],
                    reg_alpha=best_params['reg_alpha'],
                    n_jobs=-1,
                    random_state=42)

metrics_xgboost = fit_n_times(xgb, train_X, train_y, test_X, test_y)

### Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

svm = SVC(C=10, kernel='poly', degree=2, probability=True, verbose=0, random_state=42)

svm_bag = BaggingClassifier(svm, n_estimators=200, max_features=0.9, max_samples=1.0, bootstrap_features=False,
                            bootstrap=True, n_jobs=1, verbose=0, random_state=42)

svm_bag.fit(train_X, train_y)
y_test_prob = svm_bag.predict_proba(test_X)[:, 1]
metrics_bag_svm = print_model_metrics(test_y, y_test_prob, return_metrics=True)

### SIMPLE NN
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

epochs=10

train_X_dense = train_X.toarray()
train_dataset = TensorDataset(torch.tensor(train_X_dense).float(), torch.tensor(train_y).float())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3054, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.use_deterministic_algorithms(True)
simple_nn = SimpleNN()
criterion = nn.BCELoss()
optimizer = Adam(simple_nn.parameters())

for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = simple_nn(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

test_X_dense = test_X.toarray()
with torch.no_grad():
    y_pred_prob_stack = simple_nn(torch.tensor(test_X_dense).float()).numpy()
    metrics_nn = print_model_metrics(test_y, y_pred_prob_stack, return_metrics=True)


from sklearn.base import BaseEstimator
class PyTorchClassifierWrapper(BaseEstimator):
    def __init__(self, model, epochs=10, batch_size=128):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters())

    def fit(self, X, y):
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, targets in loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, X):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(torch.tensor(X).float()).numpy()
        return np.hstack((1 - outputs, outputs))  # To mimic sklearn's predict_proba


### stacking clf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from scipy.sparse import issparse
import numpy as np

class StackingClassifier:

    def __init__(self):
        lr = SGDClassifier(loss='log_loss', alpha=0.1, penalty='elasticnet', random_state=42)
        svm = SVC(C=10, kernel='poly', degree=2, probability=True, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=250, min_samples_split=5, max_depth=15, n_jobs=-1, random_state=42)
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=1, n_jobs=-1, seed=42)
        nn_wrapper = PyTorchClassifierWrapper(SimpleNN())

        self.model_dict = {'LR': lr, 'SVM': svm, 'KNN': knn, 'RF': rf, 'XGB': xgb, 'NN': nn_wrapper}
        self.model_weights = {'LR': 0.9, 'SVM': 0.9, 'KNN': 0.75, 'RF': 0.75, 'XGB': 0.6, 'NN': 0.5}

    def fit(self, X, y):
        # Convert to dense if X is a sparse matrix
        if issparse(X):
            X = X.toarray()
        for model_name, model in self.model_dict.items():
            print(f'Training {model_name}')
            model.fit(X, y)

    def predict_proba(self, X):
        y_pred_prob = 0
        for model_name, model in self.model_dict.items():
            y_pred_prob += (model.predict_proba(X)[:, 1] * self.model_weights[model_name])
        y_pred_prob /= sum(self.model_weights.values())
        return y_pred_prob

    def optimize_weights(self, X, y):
        np.random.seed(42)

        if issparse(X):
            X = X.toarray()  # Convert sparse to dense

        def _run_voting_clf(model_weights):
            y_pred_prob = 0
            for model_name, model in self.model_dict.items():
                if model_name in model_weights:
                    y_pred_prob += (model.predict_proba(X)[:, 1] * model_weights[model_name])
                else:
                    raise ValueError(f"Model weight for {model_name} not found in the optimization space")
            y_pred_prob /= sum(model_weights.values())
            f1 = print_model_metrics(y, y_pred_prob, return_metrics=True, verbose=0)[0]
            return {'loss': -f1, 'status': STATUS_OK}

        trials = Trials()
        space = {
            'LR': hp.uniform('LR', 0, 1),
            'SVM': hp.uniform('SVM', 0, 1),
            'KNN': hp.uniform('KNN', 0, 1),
            'RF': hp.uniform('RF', 0, 1),
            'XGB': hp.uniform('XGB', 0, 1),
            'NN': hp.uniform('NN', 0, 1)  # Make sure this is included
        }

        best = fmin(
            fn=_run_voting_clf,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials
        )

        self.model_weights = best


stacking_clf = StackingClassifier()
stacking_clf.fit(train_X, train_y)
stacking_clf.optimize_weights(test_X, test_y)
y_pred_prob_stack = stacking_clf.predict_proba(test_X.toarray())

np.save('../data/prd_FOCH.npy', y_pred_prob_stack)  # save predictions
metrics_stack = print_model_metrics(test_y, y_pred_prob_stack, confusion=True, return_metrics=True)

filename = '../data/best_model.sav'
pickle.dump(stacking_clf, open(filename, 'wb'))

###############
# all metrics #
###############
metrics_all = pd.DataFrame(np.stack([metrics_lr,
                                     metrics_svm,
                                     metrics_knn,
                                     metrics_rf,
                                     metrics_xgboost,
                                     metrics_bag_svm,
                                     metrics_stack,
                                     metrics_nn
                                     ]),
                           columns=["f1score", "precision", "recall", "AUC", "accuracy"],
                           index=["LLR", "SVM", "KNN", "RF", "XGBoost","Bagging", "Stacking", "NN"])
metrics=metrics_all.round(decimals=3).sort_values(by="f1score", ascending=False)*100
print(metrics.to_latex(index=True, formatters={
    'f1score': "{:.1f}".format,
    'precision': "{:.1f}".format,
    'recall': "{:.1f}".format,
    'AUC': "{:.1f}".format,
    'accuracy': "{:.1f}".format
}))


model_weights_df = pd.DataFrame.from_dict(stacking_clf.model_weights,
                                          orient='index',
                                          columns=["Model Weights"])
model_weights_df['Model Weights'] *= 100
sorted_weights_df = model_weights_df.sort_values(by="Model Weights", ascending=False).round(decimals=3)
print(sorted_weights_df.to_latex(index=True, formatters={'Model Weights': "{:.1f}".format}))


#########################
# monetary implications #
#########################
import pandas as pd

s1 = pd.Series(y_pred_prob_stack, name='prdct_prb')
s2 = df_new.iloc[test_idx]['declared_income_final'].reset_index(drop=True)
df_both = pd.concat([s1, s2], axis=1)
df_both.dropna(inplace=True)

("In the experimental setting, individuals who reported an income of "
 "$<1000$ and were caught would not only have to repay the true amount of "
 "income earned (1000 ECU) but also a fine that is as high as the cheated sum "
 "(in this example 1000 ECU). ")


random=sum(abs(df_both["declared_income_final"].sample(n=103) - 1000))*2 # random sample
print(random)
clf=sum(abs(df_both.nsmallest(103, 'prdct_prb')["declared_income_final"] - 1000))*2 # clf selection
print(clf)
perfect_information=sum(abs(df_both.loc[df_both["declared_income_final"] == 0].sample(n=103)["declared_income_final"] - 1000))*2
        # getting the worst cheaters
print((clf-random)/random*100) #percentage increase