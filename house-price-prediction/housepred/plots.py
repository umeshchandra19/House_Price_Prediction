"""Charts used in the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})
BLUE, GREEN, ORANGE, GREY, RED = "#2f6db5", "#1b7f4c", "#d98a1c", "#888888", "#c0392b"


def fig_model_comparison(comp, path):
    order = ["Linear Regression (raw features)", "Linear Regression", "Decision Tree (untuned)",
             "Decision Tree (tuned)", "Random Forest (extra)"]
    d = comp.set_index("model").loc[order]
    labels = [o.replace(" (", "\n(") for o in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(x - 0.19, d["cv_r2_mean"], 0.38, yerr=d["cv_r2_std"], color=BLUE, label="cross-validation (training data)", capsize=2)
    ax[0].bar(x + 0.19, d["test_r2"], 0.38, color=GREEN, label="held-out test data")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("R-squared (share of price variation explained)")
    ax[0].set_title("(a) How much of the price each model explains")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax[1].bar(x, d["test_rmse"] / 1000, color=ORANGE)
    for xi, v in zip(x, d["test_rmse"] / 1000):
        ax[1].text(xi, v + 0.8, f"${v:.0f}k", ha="center", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("Typical error on the test data ($ thousands)")
    ax[1].set_title("(b) Typical size of the mistake")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pred_vs_actual(y, p_lr, p_dt, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    lim = [0, 520_000]
    for a, p, t in [(ax[0], p_lr, "Linear Regression"), (ax[1], p_dt, "Decision Tree (tuned)")]:
        a.hexbin(y / 1000, p / 1000, gridsize=45, cmap="Blues", bins="log", mincnt=1, extent=(0, 520, 0, 520))
        a.plot([0, 520], [0, 520], color=RED, lw=1)
        a.set_title(t)
        a.set_xlabel("True price ($ thousands)")
    ax[0].set_ylabel("Predicted price ($ thousands)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_depth_curve(curve, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(curve["max_depth"], curve["train_r2"], marker="o", ms=3, color=ORANGE, label="training data")
    ax.plot(curve["max_depth"], curve["cv_r2"], marker="o", ms=3, color=BLUE, label="cross-validation (new data)")
    best = curve.loc[curve["cv_r2"].idxmax()]
    ax.axvline(best["max_depth"], color=GREY, ls="--", lw=0.8)
    ax.text(best["max_depth"] + 0.3, ax.get_ylim()[1] - 0.06, f"best depth = {int(best['max_depth'])}", fontsize=8, color=GREY)
    ax.set_xlabel("How deep the tree is allowed to grow")
    ax.set_ylabel("R-squared")
    ax.set_title("Deeper trees memorise the training data")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_importance(imp, coef, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    top = imp.head(8)[::-1]
    ax[0].barh(top.index, top.values, color=GREEN)
    ax[0].set_title("(a) What the decision tree relies on")
    ax[0].set_xlabel("Importance (adds up to 1)")
    coef = coef[[i for i in coef.index if "ISLAND" not in i]]   # only 5 island areas: too few to trust
    c = coef.reindex(coef.abs().sort_values(ascending=False).index).head(8)[::-1]
    ax[1].barh(c.index, c.values / 1000, color=[BLUE if v > 0 else RED for v in c.values])
    ax[1].set_title("(b) What the linear model uses")
    ax[1].set_xlabel("Price change ($ thousands) per one-step rise")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
