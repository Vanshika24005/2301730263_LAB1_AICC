"""
Generate a SYNTHETIC login-attempt dataset (data/login_attempts.csv).

Why synthetic? It keeps the project self-contained and reproducible, and lets us
control a known bias source for Task 5. It is NOT real traffic: results show how the
methods behave, not how they would perform in production.

Design notes
 * ~10% of events are suspicious (class imbalance, as in real life).
 * 25% of attacks are 'low-and-slow' and look almost normal (hard cases).
 * Region B is a small group of legitimate users with night-shift work patterns,
   frequent VPN use and shared IP ranges. These are legitimate behaviours that
   *look* suspicious to a naive model -> used for the bias experiment in Task 5.
 * ~3% of values are blanked in three columns to exercise missing-value handling.
"""
import numpy as np
import pandas as pd
from common import DATA_PATH, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)
N = 10_000

region = rng.choice(["A", "B"], size=N, p=[0.88, 0.12])
label = (rng.random(N) < 0.10).astype(int)
stealth = (label == 1) & (rng.random(N) < 0.25)       # low-and-slow attacks

rows = {}
rows["user_region"] = region
rows["label"] = label

# hour of day
hour = np.where(label == 1,
                np.where(rng.random(N) < 0.55, rng.normal(2.5, 1.8, N), rng.uniform(0, 24, N)),
                np.where((region == "B") & (rng.random(N) < 0.5),
                         rng.normal(2.5, 2.5, N), rng.normal(11.5, 3.5, N)))
hour = np.round(hour) % 24
hour = np.where(stealth, np.round(rng.normal(11.5, 3.5, N)) % 24, hour)
rows["hour_of_day"] = hour.astype(int)

# failed attempts before success / event
fa = np.where(label == 1, rng.poisson(4.5, N), rng.poisson(0.4, N))
fa = np.where(stealth, rng.poisson(1.2, N), fa)
rows["failed_attempts"] = fa

# IP reputation (0 good - 100 bad)
ip = np.where(label == 1, rng.normal(62, 20, N),
              np.where(region == "B", rng.normal(28, 12, N), rng.normal(15, 10, N)))
ip = np.where(stealth, rng.normal(30, 12, N), ip)
rows["ip_reputation_score"] = np.clip(ip, 0, 100).round(1)

# requests per minute
rpm = np.where(label == 1, rng.lognormal(np.log(55), 0.5, N), rng.lognormal(np.log(11), 0.5, N))
rpm = np.where(stealth, rng.lognormal(np.log(16), 0.5, N), rpm)
rows["requests_per_min"] = rpm.round(1)

# session duration (seconds)
sd = np.where(label == 1, rng.normal(120, 90, N), rng.normal(600, 250, N))
sd = np.where(stealth, rng.normal(420, 200, N), sd)
rows["session_duration_s"] = np.clip(sd, 5, None).round(0)

# data transferred (KB)
kb = np.where(label == 1, rng.lognormal(np.log(4000), 1.0, N), rng.lognormal(np.log(600), 0.9, N))
kb = np.where(stealth, rng.lognormal(np.log(900), 0.9, N), kb)
rows["bytes_transferred_kb"] = kb.round(0)

# VPN usage
p_vpn = np.where(label == 1, 0.50, np.where(region == "B", 0.35, 0.05))
rows["uses_vpn"] = (rng.random(N) < p_vpn).astype(int)

# country (abstract codes, C5 is a rare origin)
c_legit_A = rng.choice(["C1", "C2", "C3", "C4", "C5"], N, p=[0.45, 0.25, 0.15, 0.13, 0.02])
c_legit_B = rng.choice(["C1", "C2", "C3", "C4", "C5"], N, p=[0.05, 0.05, 0.10, 0.10, 0.70])
c_attack = rng.choice(["C1", "C2", "C3", "C4", "C5"], N, p=[0.10, 0.10, 0.15, 0.15, 0.50])
rows["country"] = np.where(label == 1, c_attack, np.where(region == "B", c_legit_B, c_legit_A))

# device and authentication
dev = np.where(label == 1, rng.choice(["desktop", "mobile", "unknown"], N, p=[0.3, 0.2, 0.5]),
               rng.choice(["desktop", "mobile", "unknown"], N, p=[0.60, 0.37, 0.03]))
rows["device_type"] = dev
auth = np.where(label == 1, rng.choice(["password", "mfa", "sso"], N, p=[0.85, 0.10, 0.05]),
                rng.choice(["password", "mfa", "sso"], N, p=[0.40, 0.35, 0.25]))
rows["auth_method"] = auth

df = pd.DataFrame(rows)
# inject missing values
for col in ["ip_reputation_score", "session_duration_s", "device_type"]:
    df.loc[rng.random(N) < 0.03, col] = np.nan

df.insert(0, "event_id", np.arange(1, N + 1))
df.to_csv(DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {DATA_PATH}")
print(df["label"].value_counts().rename({0: "normal", 1: "suspicious"}))
