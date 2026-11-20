"""Task 2: Exploratory data analysis and threat-pattern visualisation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import OUT, load_data, NUM_FEATURES, save_json

sns.set_theme(style="whitegrid")
df = load_data()
df["class"] = df["label"].map({0: "Normal", 1: "Suspicious"})
palette = {"Normal": "#4C72B0", "Suspicious": "#C44E52"}

summary = {
    "rows": int(len(df)),
    "suspicious_share_pct": round(df["label"].mean() * 100, 2),
    "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
}
print(summary)

# 1. Class balance
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="class", hue="class", palette=palette, legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
ax.set_title("Normal vs suspicious events"); ax.set_xlabel(""); ax.set_ylabel("Events")
fig.tight_layout(); fig.savefig(OUT / "eda_class_balance.png", dpi=150); plt.close(fig)

# 2. Distribution of attack-related features
feats = ["failed_attempts", "ip_reputation_score", "requests_per_min",
         "session_duration_s", "bytes_transferred_kb", "hour_of_day"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), feats):
    data = df.copy()
    if f in ("requests_per_min", "bytes_transferred_kb"):
        sns.histplot(data=data, x=f, hue="class", palette=palette, bins=40,
                     log_scale=True, stat="density", common_norm=False, ax=ax)
    else:
        sns.histplot(data=data, x=f, hue="class", palette=palette, bins=30,
                     stat="density", common_norm=False, ax=ax)
    ax.set_title(f)
fig.suptitle("Feature distributions: normal vs suspicious", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "eda_feature_distributions.png", dpi=150); plt.close(fig)

# 3. Frequency of suspicious activity by hour, country, authentication method
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
by_hour = df.groupby("hour_of_day")["label"].mean() * 100
axes[0].bar(by_hour.index, by_hour.values, color="#C44E52")
axes[0].set_title("Suspicious rate by hour of day (%)"); axes[0].set_xlabel("hour")
by_c = df.groupby("country")["label"].mean() * 100
axes[1].bar(by_c.index, by_c.values, color="#DD8452")
axes[1].set_title("Suspicious rate by country code (%)")
by_a = df.groupby("auth_method")["label"].mean() * 100
axes[2].bar(by_a.index, by_a.values, color="#8172B3")
axes[2].set_title("Suspicious rate by auth method (%)")
fig.tight_layout(); fig.savefig(OUT / "eda_suspicious_frequency.png", dpi=150); plt.close(fig)

# 4. Failed attempts frequency
fig, ax = plt.subplots(figsize=(6, 4))
fa = df.assign(fa=df["failed_attempts"].clip(upper=10))
sns.countplot(data=fa, x="fa", hue="class", palette=palette, ax=ax)
ax.set_yscale("log"); ax.set_xlabel("failed attempts (10 = 10+)"); ax.set_ylabel("events (log)")
ax.set_title("Failed login attempts")
fig.tight_layout(); fig.savefig(OUT / "eda_failed_attempts.png", dpi=150); plt.close(fig)

# 5. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[NUM_FEATURES + ["label"]].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig(OUT / "eda_correlation.png", dpi=150); plt.close(fig)

# Threat-indicator table (mean per class) + correlation with label
ind = df.groupby("class")[NUM_FEATURES].mean().T.round(2)
ind["corr_with_label"] = df[NUM_FEATURES + ["label"]].corr()["label"].drop("label").round(2)
ind.to_csv(OUT / "threat_indicators.csv")
summary["indicators"] = ind.reset_index().rename(columns={"index": "feature"}).to_dict("records")
save_json("eda_summary.json", summary)
print(ind)
