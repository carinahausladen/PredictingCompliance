# [Anirudh Shenoy, 2019](https://towardsdatascience.com/text-classification-with-extremely-small-datasets-333d322caee2)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk import word_tokenize
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

sns.set_palette("muted")


def calc_f1(p_and_r):
    p, r = p_and_r
    if p == 0 and r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics
# Since we are optimizing for F1 score - we will first calculate precision and recall and
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob, confusion=False, verbose=True, return_metrics=False):
    precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label=1)

    # Find the threshold value that gives the best F1 Score
    best_f1_index = np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[
        best_f1_index]

    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)
    # print(y_test_pred)

    # Calculate all metrics
    f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary')
    roc_auc = roc_auc_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)

    if confusion:
        # Calculate and Display the confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0)  # for label size
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No Clickbait', 'Clickbait'],
                    yticklabels=['No Clickbait', 'Clickbait'], annot_kws={"size": 14}, cmap='Blues')  # font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')

    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision,
                                                                                                best_recall, roc_auc,
                                                                                                acc))
    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc])


# Run Simple Log Reg Model and Print metrics
from sklearn.linear_model import SGDClassifier


# Run log reg 10 times and average the result to reduce prediction variance
def run_log_reg(train_features, test_features, y_train, y_test, alpha=1e-4, confusion=False, return_f1=False,
                verbose=True):
    metrics = list()
    for _ in range(10):
        log_reg = SGDClassifier(loss='log', alpha=alpha, n_jobs=-1, penalty='l2')
        log_reg.fit(train_features, y_train)
        y_test_prob = log_reg.predict_proba(test_features)[:, 1]
        metrics.append(
            print_model_metrics(y_test, y_test_prob, confusion=confusion, verbose=False, return_metrics=True))
    metrics_matrix = np.stack(metrics)  # 10x5 matrix
    metrics = np.mean(metrics_matrix, axis=0)
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return f1_score
    return metrics, log_reg, metrics_matrix


def adjusted_f1(y_true, y_prob):
    f1 = print_model_metrics(y_true, y_prob, verbose=0, return_metrics=True)[0]
    return f1


score = make_scorer(adjusted_f1, greater_is_better=True, needs_proba=True)


def run_grid_search(model, params, x_train, y_train):
    grid = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring=score, verbose=0, refit=False)
    grid.fit(x_train, y_train)
    return (grid.best_params_, grid.best_score_)


def fit_n_times(model, x_train, y_train, x_test, y_test, n_iters=10):
    metrics = np.zeros(5)
    for _ in range(n_iters):
        model.fit(x_train, y_train)
        y_test_prob = model.predict_proba(x_test)[:, 1]
        metrics += print_model_metrics(y_test, y_test_prob, verbose=False, return_metrics=True)
    metrics /= 10
    print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    return metrics


def tfdf_embdngs(documents, embedings, dict_tf):
    vectors = []
    for title in tqdm(documents):
        w2v_vectors = embedings.query(word_tokenize(title))
        weights = [dict_tf.get(word, 1) for word in word_tokenize(title)]
        vectors.append(np.average(w2v_vectors, axis=0, weights=weights))
    return np.array(vectors)
