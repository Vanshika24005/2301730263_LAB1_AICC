"""
Task 5: Ethical-risk simulation.
 (a) Threshold sweep: false positives block legitimate users.
 (b) Fairness audit: false-positive rate (FPR) for Region A vs Region B legitimate users.
 (c) Biased training data: under-represent Region B in training and measure the effect.
 (d) Mitigation: drop proxy features (country, VPN) and observe the accuracy/fairness trade-off.
 (e) Explainability: coefficients (transparent) vs permutation importance (black-box MLP).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from common import (OUT, TARGET, GROUP, NUM_FEATURES, CAT_FEATURES, RANDOM_STATE,
                    build_preprocessor, load_data, save_json, split_data)

df = load_data()
train, test = split_data(df)
FEATS = NUM_FEATURES + CAT_FEATURES
yte = test[TARGET].values


def make_lr(feats_num=NUM_FEATURES, feats_cat=CAT_FEATURES, **kw):
    return Pipeline([("prep", build_preprocessor(feats_num, feats_cat)),
                     ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kw))])


def pipe_feats(pipe):
    return [c for t in pipe.named_steps["prep"].transformers_ for c in t[2]]


def audit(pipe, data, thr=0.5):
    """Return overall + per-group stats at a given decision threshold."""
    proba = pipe.predict_proba(data[pipe_feats(pipe)])[:, 1]
    pred = (proba >= thr).astype(int)
    y = data[TARGET].values
    g = data[GROUP].values
    out = {}
    for name, mask in [("All", np.ones_like(y, bool)), ("Region A", g == "A"), ("Region B", g == "B")]:
        legit = mask & (y == 0)
        att = mask & (y == 1)
        flagged = mask & (pred == 1)
        out[name] = {
            "fpr_pct": float(pred[legit].mean() * 100),
            "legit_blocked": int(pred[legit].sum()),
            "legit_total": int(legit.sum()),
            "recall_pct": float(pred[att].mean() * 100) if att.sum() else None,
            "precision_pct": float(y[flagged].mean() * 100) if flagged.sum() else None,
        }
    return out


# ---------- (a) threshold sweep ----------
base = make_lr().fit(train[FEATS], train[TARGET])
proba = base.predict_proba(test[FEATS])[:, 1]
rows = []
for thr in np.arange(0.05, 0.96, 0.05):
    pred = (proba >= thr).astype(int)
    tp = ((pred == 1) & (yte == 1)).sum(); fp = ((pred == 1) & (yte == 0)).sum()
    fn = ((pred == 0) & (yte == 1)).sum(); tn = ((pred == 0) & (yte == 0)).sum()
    rows.append({"threshold": round(float(thr), 2), "tp": int(tp), "fp": int(fp), "fn": int(fn),
                 "tn": int(tn), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                 "fpr": fp / (fp + tn)})
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(sweep.threshold, sweep.precision, label="precision")
axes[0].plot(sweep.threshold, sweep.recall, label="recall")
axes[0].set_xlabel("decision threshold"); axes[0].legend(); axes[0].set_title("Precision / recall trade-off")
axes[1].bar(sweep.threshold, sweep.fp, width=0.04, color="#C44E52", label="legit users blocked (FP)")
axes[1].bar(sweep.threshold, sweep.fn, width=0.02, color="#4C72B0", label="attacks missed (FN)")
axes[1].set_xlabel("decision threshold"); axes[1].legend(); axes[1].set_title("Cost of each threshold")
fig.tight_layout(); fig.savefig(OUT / "ethics_threshold_sweep.png", dpi=150); plt.close(fig)

# ---------- (b) fairness audit at several thresholds ----------
fair = {f"thr_{t}": audit(base, test, t) for t in (0.2, 0.5, 0.8)}

# ---------- (c) biased training data ----------
rng = np.random.default_rng(RANDOM_STATE)
keep = ~((train[GROUP] == "B") & (rng.random(len(train)) < 0.90))   # drop 90% of Region B rows
train_biased = train[keep]
biased = make_lr().fit(train_biased[FEATS], train_biased[TARGET])
biased_audit = audit(biased, test, 0.5)
balanced = make_lr(class_weight="balanced").fit(train[FEATS], train[TARGET])
balanced_audit = audit(balanced, test, 0.5)

# ---------- (d) mitigation: remove proxy features ----------
num_m = [f for f in NUM_FEATURES if f != "uses_vpn"]
cat_m = [f for f in CAT_FEATURES if f != "country"]
mitig = make_lr(num_m, cat_m).fit(train[num_m + cat_m], train[TARGET])
mitig_audit = audit(mitig, test, 0.5)

labels = ["Baseline", "Biased training\n(Region B -90%)", "class_weight\nbalanced", "Proxy features\nremoved"]
audits = [fair["thr_0.5"], biased_audit, balanced_audit, mitig_audit]
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))
ax.bar(x - 0.2, [a["Region A"]["fpr_pct"] for a in audits], 0.4, label="Region A", color="#4C72B0")
ax.bar(x + 0.2, [a["Region B"]["fpr_pct"] for a in audits], 0.4, label="Region B", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("False-positive rate on legitimate users (%)"); ax.legend()
ax.set_title("Who gets wrongly blocked? FPR by region (threshold 0.5)")
fig.tight_layout(); fig.savefig(OUT / "ethics_fairness_fpr.png", dpi=150); plt.close(fig)

# ---------- (e) explainability ----------
mlp = Pipeline([("prep", build_preprocessor()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                      max_iter=300, random_state=RANDOM_STATE))]).fit(train[FEATS], train[TARGET])
pi = permutation_importance(mlp, test[FEATS], test[TARGET], n_repeats=5,
                            random_state=RANDOM_STATE, scoring="f1")
imp = sorted(zip(FEATS, pi.importances_mean), key=lambda t: -t[1])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh([t[0] for t in imp][::-1], [t[1] for t in imp][::-1], color="#8172B3")
ax.set_title("MLP permutation importance (drop in F1)\nglobal view only - no per-decision reason")
fig.tight_layout(); fig.savefig(OUT / "ethics_mlp_importance.png", dpi=150); plt.close(fig)

res = {"fairness_by_threshold": fair, "biased_training": biased_audit,
       "balanced_class_weight": balanced_audit, "proxy_features_removed": mitig_audit,
       "mlp_permutation_importance": [(n, float(v)) for n, v in imp],
       "region_B_share_of_train_pct": float((train[GROUP] == "B").mean() * 100),
       "region_B_share_of_biased_train_pct": float((train_biased[GROUP] == "B").mean() * 100)}
save_json("ethics_metrics.json", res)
for k in ("biased_training", "balanced_class_weight", "proxy_features_removed"):
    print(k, {g: round(v["fpr_pct"], 2) for g, v in res[k].items()})
print("baseline", {g: round(v["fpr_pct"], 2) for g, v in fair["thr_0.5"].items()})
