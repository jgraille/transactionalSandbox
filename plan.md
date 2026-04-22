# Fraud Labeling Plan

**Objective:** Add a synthetic `IsFraud` column to `bank_transactions_data_2_augmented_clean_2.csv` using a multi-layered, rule-based approach grounded in real-world fraud detection heuristics.

**Target fraud rate:** 3–5% of all transactions (~1,500–2,500 out of 50,000). This matches realistic bank fraud incidence and ensures meaningful class imbalance for downstream modeling without being so rare that rules become arbitrary.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Strategy Overview](#2-strategy-overview)
3. [Phase 1 — Data Loading & Cleaning](#3-phase-1--data-loading--cleaning)
4. [Phase 1b — Column Coverage Analysis](#4-phase-1b--column-coverage-analysis)
5. [Phase 2 — Feature Engineering for Labeling](#5-phase-2--feature-engineering-for-labeling)
6. [Phase 3 — Rule Definitions](#6-phase-3--rule-definitions)
7. [Phase 4 — Scoring & Thresholding](#7-phase-4--scoring--thresholding)
8. [Phase 5 — Controlled Noise Injection](#8-phase-5--controlled-noise-injection)
9. [Phase 6 — Validation & Sanity Checks](#9-phase-6--validation--sanity-checks)
10. [Phase 7 — Export](#10-phase-7--export)
11. [Implementation Notes](#11-implementation-notes)
12. [Implementation Todo List](#12-implementation-todo-list)
13. [Appendix — Rule Rationale](#13-appendix--rule-rationale)

---

## 1. Prerequisites

Install the required packages into the `baTr` venv:

```bash
source bankTransactions/baTr/bin/activate
pip install pandas numpy scikit-learn
```

All code in this plan targets **Python 3.11** and uses only **pandas** and **numpy** (plus stdlib). Scikit-learn is needed later for modeling but not for the labeling itself.

---

## 2. Strategy Overview

Since the dataset has no ground-truth labels, we construct synthetic fraud labels by combining **two complementary approaches**:

### Approach A — Deterministic Rule-Based Flags

A set of domain-informed rules, each inspired by well-known fraud patterns (account takeover, card-not-present fraud, structuring, etc.). Each rule contributes a binary signal. Transactions that trigger **multiple rules simultaneously** are the strongest fraud candidates.

### Approach B — Statistical Outlier Scoring

Transactions that are statistical outliers relative to their own account's behavioral baseline (e.g., amount far above the account's median, unusual location for the account) get additional fraud score. This catches anomalies that no single rule would flag.

### Combining A + B

Each transaction gets a **fraud score** (0–100 scale). A threshold is applied to produce the binary `IsFraud` label. The threshold is calibrated to hit the target fraud rate.

```
┌─────────────────────────┐     ┌──────────────────────────┐
│  Deterministic Rules     │     │  Statistical Outliers     │
│  (binary flags × weight) │     │  (z-scores, percentiles)  │
└────────────┬────────────┘     └────────────┬─────────────┘
             │                                │
             ▼                                ▼
         ┌───────────────────────────────────────┐
         │   Weighted Fraud Score (0–100)         │
         └──────────────────┬────────────────────┘
                            │
                   threshold (e.g., 65)
                            │
                    ┌───────▼───────┐
                    │   IsFraud     │
                    │   (0 or 1)    │
                    └───────────────┘
```

---

## 3. Phase 1 — Data Loading & Cleaning

Handle the dual datetime format identified in `research.md` and prepare the dataframe.

```python
import pandas as pd
import numpy as np

DATA_PATH = "bank_transactions_data_2_augmented_clean_2.csv"

df = pd.read_csv(DATA_PATH)

# Normalize the dual date format (some rows have "M/D/YYYY H:MM", others "M/D/YYYY")
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="mixed", dayfirst=False)

# Extract time components (hour will be NaT-safe: NaN where time was absent)
df["Hour"] = df["TransactionDate"].dt.hour
df["DayOfWeek"] = df["TransactionDate"].dt.dayofweek  # 0=Monday, 6=Sunday
df["Month"] = df["TransactionDate"].dt.month
df["Year"] = df["TransactionDate"].dt.year

# Rename the space-containing column for ergonomics
df = df.rename(columns={"IP Address": "IPAddress"})

# Sort by account + date for temporal features
df = df.sort_values(["AccountID", "TransactionDate"]).reset_index(drop=True)
```

---

## 4. Phase 1b — Column Coverage Analysis

Before engineering any features, inspect the completeness of every column. This determines which columns are safe to use unconditionally in rules vs. which need null-aware handling.

### 4.1 Non-Null Coverage Report

```python
total_rows = len(df)
print(f"Total rows: {total_rows:,}\n")
print(f"{'Column':<25} {'Non-Null':>10} {'Coverage':>10}")
print("-" * 47)
for col in df.columns:
    non_null = df[col].notna().sum()
    pct = non_null / total_rows * 100
    flag = "" if pct == 100 else "  "
    print(f"{col:<25} {non_null:>10,} {pct:>9.2f}%{flag}")
```

Expected output pattern (based on `research.md` findings):

| Column | Expected Coverage | Notes |
|---|---|---|
| `TransactionID` | 100% | Primary key, always present |
| `AccountID` | 100% | |
| `TransactionAmount` | 100% | |
| `TransactionDate` | 100% | All rows have a date string, but the parsed datetime is always valid |
| `TransactionType` | 100% | |
| `Location` | 100% | |
| `DeviceID` | 100% | |
| `IPAddress` | 100% | (renamed from `IP Address`) |
| `MerchantID` | 100% | |
| `Channel` | 100% | |
| `CustomerAge` | 100% | |
| `CustomerOccupation` | 100% | |
| `TransactionDuration` | 100% | |
| `LoginAttempts` | 100% | |
| `AccountBalance` | 100% | |
| `Hour` | **100%** | Derived column — pandas defaults date-only rows to `00:00:00`, so `Hour=0` for those rows. No NaN values. |
| `DayOfWeek` | 100% | Derived from date portion, always available |
| `Month` | 100% | Derived from date portion, always available |
| `Year` | 100% | Derived from date portion, always available |

### 4.2 `Hour` Coverage — No Imputation Required

`Hour` has **100% coverage** — no imputation is needed. When `pd.to_datetime(..., format="mixed")` parses a date-only string (e.g. `5/19/2024`), pandas defaults the missing time component to `00:00:00`, so `dt.hour` returns `0` (midnight) rather than `NaN`. The ~25k rows that originally had no time component all get `Hour=0`.

Confirmed from running cell 4: `Hour` has exactly **4 unique values**: `0`, `16`, `17`, `18`.

All rules (R1–R11) operate on fully-covered data. The `notna()` guard on R10 is a no-op but harmless.

### 4.3 Uniqueness & Cardinality Summary

Also useful before feature engineering — understanding how many distinct values each column has:

```python
print(f"\n{'Column':<25} {'Unique':>10} {'Dtype':<15}")
print("-" * 52)
for col in df.columns:
    print(f"{col:<25} {df[col].nunique():>10,} {str(df[col].dtype):<15}")
```

This confirms:
- ID columns (`TransactionID`, `AccountID`, `DeviceID`, `IPAddress`, `MerchantID`) have the expected cardinality ranges
- Categorical columns (`TransactionType`, `Channel`, `CustomerOccupation`) have low cardinality (2, 3, and 4 respectively)
- No unexpected single-value (constant) columns that would be useless as features

---

## 5. Phase 2 — Feature Engineering for Labeling

Build the derived features that the rules will operate on. These are **labeling-time features**, not modeling-time features — we compute them across the full dataset since we're constructing labels, not predicting.

### 5.1 Account-Level Aggregates

```python
acct = df.groupby("AccountID").agg(
    acct_mean_amount=("TransactionAmount", "mean"),
    acct_std_amount=("TransactionAmount", "std"),
    acct_median_amount=("TransactionAmount", "median"),
    acct_txn_count=("TransactionID", "count"),
    acct_unique_locations=("Location", "nunique"),
    acct_unique_devices=("DeviceID", "nunique"),
    acct_unique_ips=("IPAddress", "nunique"),
    acct_unique_merchants=("MerchantID", "nunique"),
).reset_index()

# Fill std=0 for accounts with only 1 transaction (avoid division by zero)
acct["acct_std_amount"] = acct["acct_std_amount"].fillna(0).replace(0, 1)

df = df.merge(acct, on="AccountID", how="left")
```

### 5.2 IP and Device Sharing Metrics

```python
# How many distinct accounts use this IP?
ip_sharing = df.groupby("IPAddress")["AccountID"].nunique().reset_index()
ip_sharing.columns = ["IPAddress", "ip_acct_count"]

# How many distinct accounts use this device?
dev_sharing = df.groupby("DeviceID")["AccountID"].nunique().reset_index()
dev_sharing.columns = ["DeviceID", "dev_acct_count"]

df = df.merge(ip_sharing, on="IPAddress", how="left")
df = df.merge(dev_sharing, on="DeviceID", how="left")
```

### 5.3 Per-Transaction Statistical Deviation

```python
# Z-score of this transaction's amount relative to the account's own history
df["amount_zscore"] = (
    (df["TransactionAmount"] - df["acct_mean_amount"]) / df["acct_std_amount"]
)

# Ratio of transaction amount to account balance
df["amount_to_balance_ratio"] = df["TransactionAmount"] / df["AccountBalance"].replace(0, 1)
```

### 5.4 Velocity Features

```python
# Time since previous transaction for the same account
df["prev_txn_date"] = df.groupby("AccountID")["TransactionDate"].shift(1)
df["hours_since_prev"] = (
    (df["TransactionDate"] - df["prev_txn_date"]).dt.total_seconds() / 3600
)

# Number of transactions by this account in the same calendar day
df["daily_txn_count"] = df.groupby(
    ["AccountID", df["TransactionDate"].dt.date]
)["TransactionID"].transform("count")
```

### 5.5 Location Consistency

```python
# Is this location one the account has used before? (minority location = suspicious)
acct_location_freq = (
    df.groupby(["AccountID", "Location"])["TransactionID"]
    .count()
    .reset_index()
    .rename(columns={"TransactionID": "acct_loc_freq"})
)
df = df.merge(acct_location_freq, on=["AccountID", "Location"], how="left")

# For each account, what fraction of their transactions happen at this location?
df["acct_loc_pct"] = df["acct_loc_freq"] / df["acct_txn_count"]
```

---

## 6. Phase 3 — Rule Definitions

Each rule returns a binary flag (0 or 1) and has an assigned weight reflecting how strongly it indicates fraud. Rules are designed to be individually weak but collectively strong.

### Rule Table

| # | Rule Name | Condition | Weight | Rationale |
|---|-----------|-----------|--------|-----------|
| R1 | High login attempts | `LoginAttempts >= 3` | 25 | Account takeover / credential stuffing |
| R2 | Very high login attempts | `LoginAttempts >= 5` | 15 | Adds on top of R1 for extreme cases |
| R3 | Amount outlier (account) | `amount_zscore > 2.5` | 20 | Abnormally large transaction for this account |
| R4 | Amount exceeds balance | `amount_to_balance_ratio > 0.8` | 15 | Draining the account — common in fraud |
| R5 | Shared IP across accounts | `ip_acct_count >= 3` | 15 | IP used by 3+ different accounts |
| R6 | Shared device across accounts | `dev_acct_count >= 3` | 15 | Device used by 3+ accounts |
| R7 | Rapid-fire transactions | `hours_since_prev < 0.5` (30 min) | 15 | Burst activity within short window |
| R8 | High daily frequency | `daily_txn_count >= 5` | 10 | Unusually many transactions in one day |
| R9 | Unusual location for account | `acct_loc_pct < 0.05` AND `acct_txn_count >= 10` | 10 | Transaction from a rarely-used location, for accounts with enough history |
| R10 | Night-time transaction | `Hour` is not NaN AND (`Hour < 5` OR `Hour >= 23`) | 5 | Late-night transactions (where time is available) |
| R11 | High amount on online channel | `Channel == "Online"` AND `TransactionAmount > 800` | 10 | Large online transaction — higher-risk channel |

```python
df["R01_high_login"] = (df["LoginAttempts"] >= 3).astype(int)
df["R02_very_high_login"] = (df["LoginAttempts"] >= 5).astype(int)
df["R03_amount_outlier"] = (df["amount_zscore"] > 2.5).astype(int)
df["R04_draining_balance"] = (df["amount_to_balance_ratio"] > 0.8).astype(int)
df["R05_shared_ip"] = (df["ip_acct_count"] >= 3).astype(int)
df["R06_shared_device"] = (df["dev_acct_count"] >= 3).astype(int)
df["R07_rapid_fire"] = (df["hours_since_prev"] < 0.5).astype(int)
df["R08_high_daily_freq"] = (df["daily_txn_count"] >= 5).astype(int)
df["R09_unusual_location"] = (
    (df["acct_loc_pct"] < 0.05) & (df["acct_txn_count"] >= 10)
).astype(int)
df["R10_nighttime"] = (
    df["Hour"].notna() & ((df["Hour"] < 5) | (df["Hour"] >= 23))
).astype(int)
df["R11_high_online"] = (
    (df["Channel"] == "Online") & (df["TransactionAmount"] > 800)
).astype(int)
```

---

## 7. Phase 4 — Scoring & Thresholding

### 7.1 Compute the Weighted Fraud Score

```python
WEIGHTS = {
    "R01_high_login": 25,
    "R02_very_high_login": 15,
    "R03_amount_outlier": 20,
    "R04_draining_balance": 15,
    "R05_shared_ip": 15,
    "R06_shared_device": 15,
    "R07_rapid_fire": 15,
    "R08_high_daily_freq": 10,
    "R09_unusual_location": 10,
    "R10_nighttime": 5,
    "R11_high_online": 10,
}

rule_cols = list(WEIGHTS.keys())
weight_values = np.array([WEIGHTS[c] for c in rule_cols])

df["fraud_score_raw"] = df[rule_cols].values @ weight_values

# Normalize to 0–100 scale
max_possible = weight_values.sum()  # 155
df["fraud_score"] = (df["fraud_score_raw"] / max_possible * 100).clip(0, 100)
```

### 7.2 Add Statistical Outlier Bonus

Give a small score boost to transactions that are statistical outliers on multiple continuous dimensions simultaneously, even if they don't trigger any single rule.

```python
# Bonus for multi-dimensional statistical unusualness
outlier_bonus = (
    (df["amount_zscore"].clip(0, 5) / 5 * 5) +   # up to +5 for extreme z-score
    (df["amount_to_balance_ratio"].clip(0, 2) / 2 * 5)  # up to +5 for high ratio
)
df["fraud_score"] = (df["fraud_score"] + outlier_bonus).clip(0, 100)
```

### 7.3 Calibrate the Threshold

```python
TARGET_FRAUD_RATE = 0.04  # 4% → ~2,000 flagged transactions

threshold = df["fraud_score"].quantile(1 - TARGET_FRAUD_RATE)
print(f"Threshold at {TARGET_FRAUD_RATE:.0%} fraud rate: {threshold:.2f}")

df["IsFraud"] = (df["fraud_score"] >= threshold).astype(int)

actual_rate = df["IsFraud"].mean()
print(f"Actual fraud rate: {actual_rate:.2%} ({df['IsFraud'].sum()} transactions)")
```

This quantile-based approach **guarantees** the fraud rate lands close to the target regardless of how the rules distribute scores.

---

## 8. Phase 5 — Controlled Noise Injection

Pure rule-based labels are deterministic and perfectly separable — a model could learn the exact rules instead of generalizing. Adding controlled noise makes the labels more realistic and forces models to learn patterns rather than memorize boundaries.

### 8.1 False Negatives (missed fraud)

Randomly flip a small percentage of `IsFraud=1` labels to 0. Simulates the real-world reality that some fraud goes undetected.

```python
np.random.seed(42)

fraud_indices = df[df["IsFraud"] == 1].index
fn_rate = 0.05  # 5% of fraud cases become false negatives
fn_mask = np.random.random(len(fraud_indices)) < fn_rate
df.loc[fraud_indices[fn_mask], "IsFraud"] = 0

print(f"Flipped {fn_mask.sum()} fraud labels to 0 (false negatives)")
```

### 8.2 False Positives (noise in legitimate)

Randomly flip a very small percentage of `IsFraud=0` with moderate fraud scores to 1. Simulates mislabeling in real datasets.

```python
legit_indices = df[
    (df["IsFraud"] == 0) & (df["fraud_score"] > threshold * 0.5)
].index
fp_rate = 0.005  # 0.5% of borderline legitimate cases
fp_mask = np.random.random(len(legit_indices)) < fp_rate
df.loc[legit_indices[fp_mask], "IsFraud"] = 1

print(f"Flipped {fp_mask.sum()} legit labels to 1 (false positives)")
```

---

## 9. Phase 6 — Validation & Sanity Checks

Before exporting, verify the labels make sense.

### 9.1 Fraud Rate Check

```python
print(f"Final fraud rate: {df['IsFraud'].mean():.2%}")
assert 0.02 <= df["IsFraud"].mean() <= 0.06, "Fraud rate out of expected range"
```

### 9.2 Score Distribution by Label

```python
print("\nFraud score distribution:")
print(df.groupby("IsFraud")["fraud_score"].describe())
```

Expected: `IsFraud=1` should have notably higher mean/median fraud scores than `IsFraud=0`, but with overlap (due to noise injection).

### 9.3 Rule Trigger Rates

```python
print("\nRule trigger rates (fraud vs legit):")
for col in rule_cols:
    fraud_rate = df.loc[df["IsFraud"] == 1, col].mean()
    legit_rate = df.loc[df["IsFraud"] == 0, col].mean()
    lift = fraud_rate / legit_rate if legit_rate > 0 else float("inf")
    print(f"  {col}: fraud={fraud_rate:.2%}, legit={legit_rate:.2%}, lift={lift:.1f}x")
```

Expected: every rule should have a positive lift (higher trigger rate among fraud). If any rule has lift < 1, it should be dropped or reconsidered.

### 9.4 Channel/Type Distribution in Fraud

```python
print("\nFraud by channel:")
print(pd.crosstab(df["Channel"], df["IsFraud"], normalize="index"))

print("\nFraud by transaction type:")
print(pd.crosstab(df["TransactionType"], df["IsFraud"], normalize="index"))
```

### 9.5 No Single Rule Dominates

```python
# Check that fraud is not 100% explained by any single rule
for col in rule_cols:
    coverage = df.loc[df["IsFraud"] == 1, col].mean()
    print(f"  {col} covers {coverage:.0%} of fraud")
    assert coverage < 0.95, f"Rule {col} covers too much fraud — labels are too deterministic"
```

---

## 10. Phase 7 — Export

Write the labeled dataset. Preserve the original columns plus `IsFraud`. Optionally include `fraud_score` for exploratory analysis. Drop all intermediate engineering columns.

```python
# Columns to keep in the final output
original_cols = [
    "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
    "TransactionType", "Location", "DeviceID", "IPAddress", "MerchantID",
    "Channel", "CustomerAge", "CustomerOccupation", "TransactionDuration",
    "LoginAttempts", "AccountBalance",
]

# Restore original column name
output = df[original_cols + ["IsFraud", "fraud_score"]].copy()
output = output.rename(columns={"IPAddress": "IP Address"})

# Sort back by TransactionID for consistency with the original file order
output = output.sort_values("TransactionID").reset_index(drop=True)

OUTPUT_PATH = "bank_transactions_data_2_augmented_clean_2_labeled.csv"
output.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(output)} rows to {OUTPUT_PATH}")
```

---

## 11. Implementation Notes

### Where to implement

All of the above should be implemented in `bankTransactions/prelims.ipynb`, one phase per cell group. This keeps the labeling process transparent and auditable.

### Suggested notebook cell structure

| Cell # | Content | Phase |
|--------|---------|-------|
| 1 | Imports + constants | — |
| 2 | Load CSV + clean dates | Phase 1 |
| 3 | Column coverage (non-null %) + cardinality | Phase 1b |
| 4 | Confirm `Hour` 100% coverage (no imputation needed — pandas defaults date-only to midnight) | Phase 1b |
| 5 | Account-level aggregates | Phase 2.1 |
| 6 | IP/Device sharing metrics | Phase 2.2 |
| 7 | Per-transaction stats | Phase 2.3 |
| 8 | Velocity features | Phase 2.4 |
| 9 | Location consistency | Phase 2.5 |
| 10 | Rule flag computation | Phase 3 |
| 11 | Scoring + threshold | Phase 4 |
| 12 | Noise injection | Phase 5 |
| 13 | Validation & sanity checks | Phase 6 |
| 14 | Export | Phase 7 |

### Reproducibility

- Set `np.random.seed(42)` at the top of the notebook. All randomness (noise injection) flows through this single seed.
- Pin the `TARGET_FRAUD_RATE` constant. Changing it adjusts the threshold automatically.

### Tuning levers

If the fraud rate or label quality needs adjustment, these are the knobs to turn (in priority order):

| Lever | Effect | Where |
|-------|--------|-------|
| `TARGET_FRAUD_RATE` | Directly controls how many transactions are labeled fraud | Phase 4 |
| Rule thresholds (e.g., `LoginAttempts >= 3`) | Changes which transactions score high | Phase 3 |
| Rule weights (e.g., `R01: 25`) | Changes how much each pattern contributes | Phase 4 |
| `fn_rate` / `fp_rate` | Controls label noise level | Phase 5 |
| Outlier bonus scaling | Adjusts contribution of continuous features | Phase 4 |

### Things to watch

- **R10 (nighttime) fires on all 50,000 rows** — `Hour` is 100% populated directly from `TransactionDate` parsing (date-only rows get `Hour=0`). No imputation was needed.
- **R07 (rapid-fire) depends on sort order** — make sure the dataframe is sorted by `AccountID` + `TransactionDate` before computing `hours_since_prev`.
- **Accounts with very few transactions** (1–3) will have unreliable z-scores. The `acct_std_amount` floor of 1 and the `acct_txn_count >= 10` guard on R09 mitigate this.
- **Hour imputation quality** should be visually confirmed in Phase 1b before proceeding. If the imputed distribution diverges significantly from the observed one, the conditioning groups can be adjusted (e.g., add `Month` or `TransactionType` as additional keys).

---

## 12. Implementation Todo List

A checklist of every individual task required to execute this plan, organized by phase. Work through them in order — each task depends on the ones before it within its phase.

### Prerequisites

- [x] **P-1.** Activate the `baTr` virtualenv (`source bankTransactions/baTr/bin/activate`)
- [x] **P-2.** Install `pandas` and `numpy` into the venv (`pip install pandas numpy`)
- [x] **P-3.** Install `scikit-learn` into the venv (`pip install scikit-learn`) — needed for future modeling, not for labeling
- [x] **P-4.** Add `baTr/` to `.gitignore` so the venv is not committed
- [x] **P-5.** Create a `requirements.txt` capturing the installed packages (`pip freeze > requirements.txt`)

### Phase 1 — Data Loading & Cleaning

- [x] **1-1.** Create cell 1 in `prelims.ipynb`: `import pandas as pd`, `import numpy as np`
- [x] **1-2.** Create cell 2: load the CSV with `pd.read_csv(DATA_PATH)`
- [x] **1-3.** Parse `TransactionDate` with `pd.to_datetime(..., format="mixed", dayfirst=False)` — pandas defaults date-only rows to `00:00:00`, so `Hour` is always an integer (never NaN). Rows without an original time component get `Hour=0` (midnight).
- [x] **1-4.** Extract derived time columns: `Hour`, `DayOfWeek`, `Month`, `Year`
- [x] **1-5.** Rename `IP Address` → `IPAddress`
- [x] **1-6.** Sort the dataframe by `["AccountID", "TransactionDate"]` and reset index
- [x] **1-7.** Print `df.shape` and `df.dtypes` to verify load was correct

### Phase 1b — Column Coverage Analysis

- [x] **1b-1.** Create cell 3: print non-null count and coverage % for every column
- [x] **1b-2.** Verify all 15 original columns are at 100% coverage — confirmed. `Hour` is also at 100% because pandas defaults date-only rows to `00:00:00`, giving `Hour=0`. No imputation needed.
- [x] **1b-3.** ~~Confirm `Hour` is at ~50%~~ — superseded: `Hour` is 100% covered directly from `TransactionDate` parsing. Date-only rows get `Hour=0` (midnight).
- [x] **1b-4.** ~~Conditional hour distribution lookup~~ — not needed; `Hour` has no missing values.
- [x] **1b-5.** ~~Global fallback array~~ — not needed.
- [x] **1b-6.** ~~Impute missing `Hour` values~~ — not needed; `missing_mask` is all-False.
- [x] **1b-7.** ~~Cast `Hour` to int~~ — already `int32` from `dt.hour`.
- [x] **1b-8.** ~~Print imputation count~~ — replaced with confirmation that `Hour` is 100% from parsing.
- [x] **1b-9.** ~~Observed vs imputed comparison~~ — replaced with `Hour` value distribution printout (shows 0, 16, 17, 18 as the 4 unique values).
- [x] **1b-10.** Print uniqueness & cardinality summary for all columns (nunique + dtype)
- [x] **1b-11.** Verify ID columns have expected cardinality, categoricals have low cardinality, no constant columns

### Phase 2 — Feature Engineering

- [x] **2-1.** Create cell 5: compute account-level aggregates (`acct_mean_amount`, `acct_std_amount`, `acct_median_amount`, `acct_txn_count`, `acct_unique_locations`, `acct_unique_devices`, `acct_unique_ips`, `acct_unique_merchants`)
- [x] **2-2.** Floor `acct_std_amount` to 1 for single-transaction accounts (avoid division by zero)
- [x] **2-3.** Merge account aggregates back to the main dataframe
- [x] **2-4.** Create cell 6: compute `ip_acct_count` (distinct accounts per IP)
- [x] **2-5.** Compute `dev_acct_count` (distinct accounts per device)
- [x] **2-6.** Merge IP and device sharing metrics back to the main dataframe
- [x] **2-7.** Create cell 7: compute `amount_zscore` (transaction amount z-score relative to account mean/std)
- [x] **2-8.** Compute `amount_to_balance_ratio` (transaction amount / account balance)
- [x] **2-9.** Create cell 8: compute `hours_since_prev` (time gap to previous transaction for same account)
- [x] **2-10.** Compute `daily_txn_count` (number of transactions by the account on the same calendar day)
- [x] **2-11.** Create cell 9: compute `acct_loc_freq` (how many times this account transacted at this location)
- [x] **2-12.** Compute `acct_loc_pct` (fraction of account's transactions at this location)

### Phase 3 — Rule Definitions

- [x] **3-1.** Create cell 10: compute R01 — `LoginAttempts >= 3`
- [x] **3-2.** Compute R02 — `LoginAttempts >= 5`
- [x] **3-3.** Compute R03 — `amount_zscore > 2.5`
- [x] **3-4.** Compute R04 — `amount_to_balance_ratio > 0.8`
- [x] **3-5.** Compute R05 — `ip_acct_count >= 3`
- [x] **3-6.** Compute R06 — `dev_acct_count >= 3`
- [x] **3-7.** Compute R07 — `hours_since_prev < 0.5`
- [x] **3-8.** Compute R08 — `daily_txn_count >= 5`
- [x] **3-9.** Compute R09 — `acct_loc_pct < 0.05` AND `acct_txn_count >= 10`
- [x] **3-10.** Compute R10 — `Hour < 5` OR `Hour >= 23` (with `notna()` guard)
- [x] **3-11.** Compute R11 — `Channel == "Online"` AND `TransactionAmount > 800`
- [x] **3-12.** Print trigger counts for all 11 rules to verify none are all-zero or all-one

### Phase 4 — Scoring & Thresholding

- [x] **4-1.** Create cell 11: define the `WEIGHTS` dictionary for all 11 rules
- [x] **4-2.** Compute `fraud_score_raw` via weighted dot product of rule columns
- [x] **4-3.** Normalize to `fraud_score` on a 0–100 scale
- [x] **4-4.** Compute and add the statistical outlier bonus (z-score + balance ratio components)
- [x] **4-5.** Clip final `fraud_score` to [0, 100]
- [x] **4-6.** Set `TARGET_FRAUD_RATE = 0.04`
- [x] **4-7.** Compute the threshold via `df["fraud_score"].quantile(1 - TARGET_FRAUD_RATE)`
- [x] **4-8.** Create the `IsFraud` column by applying the threshold
- [x] **4-9.** Print the threshold value, actual fraud count, and actual fraud rate

### Phase 5 — Controlled Noise Injection

- [x] **5-1.** Create cell 12: select all `IsFraud=1` indices
- [x] **5-2.** Apply false-negative flips at 5% rate (`fn_rate = 0.05`) — flip selected fraud labels to 0
- [x] **5-3.** Print how many fraud labels were flipped to 0
- [x] **5-4.** Select borderline legitimate indices (`IsFraud=0` with `fraud_score > threshold * 0.5`)
- [x] **5-5.** Apply false-positive flips at 0.5% rate (`fp_rate = 0.005`) — flip selected legit labels to 1
- [x] **5-6.** Print how many legit labels were flipped to 1

### Phase 6 — Validation & Sanity Checks

- [x] **6-1.** Create cell 13: print final fraud rate and assert it is between 2% and 6%
- [x] **6-2.** Print fraud score descriptive stats grouped by `IsFraud` (mean, std, min, max, quartiles)
- [x] **6-3.** Verify `IsFraud=1` has notably higher mean/median score than `IsFraud=0`
- [x] **6-4.** Print per-rule trigger rates for fraud vs. legit, with lift
- [x] **6-5.** Verify every rule has lift > 1 (higher trigger rate in fraud class)
- [x] **6-6.** Print fraud rate crosstab by `Channel`
- [x] **6-7.** Print fraud rate crosstab by `TransactionType`
- [x] **6-8.** For each rule, print the fraction of all fraud cases it covers
- [x] **6-9.** Assert no single rule covers >95% of fraud (no rule dominance)

### Phase 7 — Export

- [x] **7-1.** Create cell 14: define the list of original columns to keep
- [x] **7-2.** Build the output dataframe with original columns + `IsFraud` + `fraud_score`
- [x] **7-3.** Rename `IPAddress` back to `IP Address` for output consistency
- [x] **7-4.** Sort output by `TransactionID`
- [x] **7-5.** Write to `bank_transactions_data_2_augmented_clean_2_labeled.csv`
- [x] **7-6.** Print row count and filename to confirm export
- [x] **7-7.** Spot-check: reload the exported CSV and verify shape, column names, and fraud rate match expectations

---

## 13. Appendix — Rule Rationale

### R1/R2 — Login Attempts

Multiple failed login attempts before a transaction is the hallmark of **credential stuffing** or **brute-force account takeover**. In real banking data, 2+ attempts before a successful transaction correlates strongly with unauthorized access. We use a two-tier approach: moderate concern at 3+, high concern at 5+.

### R3 — Amount Outlier

A transaction far above the account's typical spending pattern suggests the account may be compromised. A z-score of 2.5 means the transaction is in the top ~0.6% of the account's own distribution — unusual enough to warrant scrutiny.

### R4 — Account Draining

When a transaction consumes >80% of the available balance, it resembles a **cash-out** or **account draining** pattern common in account takeover fraud. Legitimate customers rarely spend that close to their full balance in a single transaction.

### R5/R6 — Shared IP / Device

The same IP address or physical device being used across 3+ different accounts is a strong indicator of a **fraud ring** or **automated bot**. Legitimate sharing (e.g., a family computer) typically involves 1–2 accounts, not 3+.

### R7 — Rapid-Fire Transactions

Multiple transactions from the same account within 30 minutes suggest **automated or programmatic activity** — either a bot draining an account or card-testing behavior.

### R8 — High Daily Frequency

5+ transactions in a single day from one account is unusual for consumer banking and may indicate **structuring** (splitting a large amount into many small ones to avoid detection) or automated fraud.

### R9 — Unusual Location

A transaction from a location that represents <5% of the account's history (for accounts with 10+ transactions) suggests the account is being used in an **unfamiliar geography** — a common fraud pattern when stolen credentials are used remotely.

### R10 — Night-Time

Transactions between 11 PM and 5 AM are statistically less common for legitimate banking and slightly more common for fraud (the attacker operates outside the victim's normal schedule). This is a weak signal but adds value in combination with other flags.

### R11 — High Amount Online

Large transactions through the online channel carry higher inherent fraud risk than in-person (ATM/Branch) transactions because there is no physical authentication. The $800 threshold represents the approximate top 5% of transaction amounts in the dataset.
