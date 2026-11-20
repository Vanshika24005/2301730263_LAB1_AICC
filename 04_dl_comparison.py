"""Task 4: Neural network (MLPClassifier) vs classical ML - accuracy, time, feasibility."""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
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
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Neural Net (MLP 64-32)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                            early_stopping=True, max_iter=300,
                                            random_state=RANDOM_STATE),
}
res, mlp_pipe = {}, None
for name, clf in models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    t0 = time.perf_counter(); pipe.fit(Xtr, ytr); train_t = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = pipe.predict(Xte); pred_t = time.perf_counter() - t0
    res[name] = {"accuracy": accuracy_score(yte, pred),
                 "precision": precision_score(yte, pred),
                 "recall": recall_score(yte, pred),
                 "f1": f1_score(yte, pred),
                 "train_time_s": train_t,
                 "predict_time_ms_per_1000": pred_t / len(Xte) * 1000 * 1000}
    print(name, {k: round(v, 4) for k, v in res[name].items()})
    if name.startswith("Neural"):
        mlp_pipe = pipe

res["mlp_epochs_run"] = int(mlp_pipe.named_steps["clf"].n_iter_)
n_params = sum(w.size for w in mlp_pipe.named_steps["clf"].coefs_) + \
           sum(b.size for b in mlp_pipe.named_steps["clf"].intercepts_)
res["mlp_parameters"] = int(n_params)
save_json("dl_metrics.json", res)

# Plots: metric comparison + MLP learning curve
names = [n for n in res if isinstance(res[n], dict)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
w = 0.2
for i, m in enumerate(["accuracy", "precision", "recall", "f1"]):
    axes[0].bar([x + i * w for x in range(len(names))], [res[n][m] for n in names], w, label=m)
axes[0].set_xticks([x + 1.5 * w for x in range(len(names))])
axes[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8); axes[0].set_title("Detection metrics")
axes[1].bar(range(len(names)), [res[n]["train_time_s"] for n in names], color="#55A868")
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
axes[1].set_title("Training time (s)")
clf = mlp_pipe.named_steps["clf"]
axes[2].plot(clf.loss_curve_, label="training loss")
axes[2].plot([1 - s for s in clf.validation_scores_], label="1 - validation accuracy")
axes[2].set_xlabel("epoch"); axes[2].legend(); axes[2].set_title("MLP learning curve")
fig.tight_layout(); fig.savefig(OUT / "dl_comparison.png", dpi=150); plt.close(fig)
