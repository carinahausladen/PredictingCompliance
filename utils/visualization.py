import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def plot_decision_function(X, y, clf, ax, title=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
    if title is not None:
        ax.set_title(title)
        

def plot_decision_function_comparison():
    def create_dataset(
        n_samples=1000,
        weights=(0.01, 0.01, 0.98),
        n_classes=3,
        class_sep=0.8,
        n_clusters=1,
    ):
        return make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters,
            weights=list(weights),
            class_sep=class_sep,
            random_state=0,
        )

    X, y = create_dataset(n_samples=100, weights=(0.05, 0.25, 0.7))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    clf = LogisticRegression()
    clf.fit(X, y)
    plot_decision_function(X, y, clf, axs[0], title="Without resampling")

    sampler = RandomOverSampler(random_state=0)
    model = make_pipeline(sampler, clf).fit(X, y)
    plot_decision_function(X, y, model, axs[1], f"Using {model[0].__class__.__name__}")

    fig.suptitle(f"Decision function of {clf.__class__.__name__}")
    fig.tight_layout()