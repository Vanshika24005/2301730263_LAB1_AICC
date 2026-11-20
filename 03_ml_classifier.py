"""Task 3: Supervised ML threat classification (Logistic Regression + Decision Tree)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from common import (OUT, TARGET, NUM_FEATURES, CAT_FEATURES, RANDOM_STATE,
                    build_preprocessor, load_data, save_json, split_data)

df = load_data()
train, test = split_data(df)
FEATS = NUM_FEATURES + CAT_FEATURES
Xtr, ytr, Xte, yte = train[FEATS], train[TARGET], test[FEATS], test[TARGET]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
}
results = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, clf) in zip(axes, models.items()):
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    results[name] = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=["Normal", "Suspicious"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
    print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[name].items()
                                    if k != "confusion_matrix"))
fig.tight_layout(); fig.savefig(OUT / "ml_confusion_matrices.png", dpi=150); plt.close(fig)

# Baseline: a majority-class "model" shows why accuracy alone is misleading
base_acc = 1 - yte.mean()
results["Majority-class baseline"] = {"accuracy": base_acc, "precision": 0.0,
                                      "recall": 0.0, "f1": 0.0}
print(f"Majority-class baseline accuracy: {base_acc:.3f} (recall 0)")

# Logistic regression coefficients = simple explanation
lr = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
names = lr.named_steps["prep"].get_feature_names_out()
coef = lr.named_steps["clf"].coef_[0]
order = np.argsort(np.abs(coef))[::-1][:10]
top = [(names[i].replace("num__", "").replace("cat__", ""), float(coef[i])) for i in order]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1],
        color=["#C44E52" if t[1] > 0 else "#4C72B0" for t in top][::-1])
ax.set_title("Top logistic-regression coefficients\n(red = raises suspicion)")
fig.tight_layout(); fig.savefig(OUT / "ml_top_coefficients.png", dpi=150); plt.close(fig)
results["top_coefficients"] = top
save_json("ml_metrics.json", results)
