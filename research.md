# Research Report: transactionalFraudSandbox

**Date:** 2026-03-29  
**Scope:** Full in-depth review of the repository at `transactionalSandbox`

---

## 1. Project Overview

This is a Python-based sandbox repository whose stated purpose (from `README.md`) is:

> *"A place to test different approaches for transactional prediction purpose"*

The repository name (`transactionalSandbox`) sharpens the intent: experimenting with fraud detection or transactional risk-scoring models on bank transaction data.

**Current state:** The project is at day-zero in terms of analysis code. The only substantive content is a large pre-cleaned dataset and an empty Jupyter notebook. No EDA, feature engineering, or model code has been written yet.

---

## 2. Repository Structure

```
transactionalSandbox/
├── README.md                         # One-line project description
├── LICENSE                           # Apache License 2.0
├── .gitignore                        # Standard Python/Jupyter ignore patterns (208 lines)
├── research.md                       # This file
└── bankTransactions/
    ├── prelims.ipynb                 # Empty Jupyter notebook (entry point)
    ├── bank_transactions_data_2_augmented_clean_2.csv   # Main dataset (50k rows)
    └── baTr/                         # Local Python 3.11 virtualenv
        ├── pyvenv.cfg
        ├── bin/   (activate scripts, jupyter, ipython, pip, python symlinks)
        └── lib/python3.11/site-packages/  (full Jupyter kernel stack)
```

### Notes on structure

- **No Python source files (`.py`)** exist outside the venv.
- **No `requirements.txt`, `pyproject.toml`, or `Pipfile`** — the installed packages are only discoverable by inspecting the venv directly.
- The venv `baTr/` is **not excluded by `.gitignore`** (the file covers `venv/`, `.venv/`, `env/` etc. but not `baTr`), so it appears as untracked in git and would be committed if `git add .` were run.
- The `.gitignore` is a comprehensive modern Python template (covers pip, pytest, mypy, Ruff, Cursor, Marimo, PDM, pixi, and more).

---

## 3. License

The project is released under the **Apache License 2.0** (full text present in `LICENSE`). This is a permissive open-source license allowing use, modification, and distribution with attribution requirements but no copyleft obligation.

---

## 4. The Virtual Environment (`baTr/`)

### 4.1 Configuration

| Property | Value |
|---|---|
| Python version | **3.11.11** |
| Base interpreter | Homebrew `python@3.11` at `/opt/homebrew/opt/python@3.11/bin` |
| Created with | `python3.11 -m venv .../bankTransactions/baTr` |
| System site packages | Not included (`include-system-site-packages = false`) |

### 4.2 Installed Packages

The venv contains a **complete Jupyter kernel stack** — sufficient to run the notebook out of the box. No data science libraries (pandas, numpy, scikit-learn, matplotlib) are installed yet.

| Package | Version | Role |
|---|---|---|
| `pip` | 26.0.1 | Package manager |
| `setuptools` | 75.6.0 | Build infrastructure |
| `ipython` | **9.10.1** | Interactive Python shell |
| `ipython_pygments_lexers` | 1.1.1 | Syntax highlighting for IPython |
| `ipykernel` | **7.2.0** | Jupyter kernel backend |
| `jupyter_client` | 8.8.0 | Kernel communication protocol |
| `jupyter_core` | 5.9.1 | Core Jupyter utilities |
| `debugpy` | 1.8.20 | VS Code / Cursor debugger support |
| `comm` | 0.2.3 | Kernel-widget communication |
| `tornado` | 6.5.5 | Async networking (Jupyter server) |
| `pyzmq` | 27.1.0 | ZeroMQ bindings (kernel messaging) |
| `python_dateutil` | 2.9.0.post0 | Datetime parsing utilities |
| `traitlets` | 5.14.3 | Config/type system for Jupyter |
| `prompt_toolkit` | 3.0.52 | Rich interactive terminal UI |
| `pygments` | 2.19.2 | Syntax highlighting |
| `jedi` | 0.19.2 | Code completion engine |
| `parso` | 0.8.6 | Python parser (Jedi dependency) |
| `matplotlib_inline` | 0.2.1 | Inline plot rendering in notebooks |
| `psutil` | 7.2.2 | System/process monitoring |
| `platformdirs` | 4.9.4 | OS-appropriate directory paths |
| `packaging` | 26.0 | PEP 440 version parsing |
| `nest_asyncio` | 1.6.0 | Nested event loop support |
| `appnope` | 0.1.4 | macOS app-nap prevention |
| `pexpect` | 4.9.0 | Process spawning |
| `ptyprocess` | 0.7.0 | Pseudoterminal subprocess |
| `asttokens` | 3.0.1 | AST source position tracking |
| `executing` | 2.2.1 | Runtime code introspection |
| `stack_data` | 0.6.3 | Stack frame data extraction |
| `pure_eval` | 0.2.3 | Safe expression evaluation |
| `decorator` | 5.2.1 | Function decorator utilities |
| `wcwidth` | 0.6.0 | Unicode character width |
| `six` | 1.17.0 | Python 2/3 compat shim |
| `typing_extensions` | 4.15.0 | Backported typing features |

**Key observation:** The environment is purpose-built to run `prelims.ipynb` as a Jupyter kernel inside Cursor (the presence of `debugpy` is specifically for the VS Code/Cursor Python extension's debugger). However, **data science packages are absent** — `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` etc. all need to be installed before any analysis can begin.

---

## 5. The Notebook (`prelims.ipynb`)

- **Format:** nbformat 4 / minor 5 (current standard)
- **Language:** Python
- **Kernel metadata:** `language_info.name = "python"` — no specific kernel name is bound, meaning it will use whichever Python kernel is active in the IDE
- **Cells:** 1 empty `code` cell (`"source": []`)
- **Outputs:** None
- **Name:** "prelims" strongly implies this is intended as a preliminary / EDA notebook

The notebook is a blank canvas; nothing has been executed.

---

## 6. The Dataset

### 6.1 File

`bankTransactions/bank_transactions_data_2_augmented_clean_2.csv`

The filename encodes a processing history:
- `bank_transactions_data_2` — second iteration of a base dataset
- `augmented` — enriched or synthetically expanded
- `clean` — has gone through a cleaning pass
- `_2` — second version of the clean output

The augmentation and cleaning scripts are **not present** in this repo; only the final output CSV is included.

### 6.2 Dimensions

| Property | Value |
|---|---|
| Rows (transactions) | **50,000** |
| Columns | **15** |
| File format | CSV, comma-delimited, UTF-8 |
| Header row | Yes |
| Missing values | None observed in sampled rows |
| Fraud/label column | **None** |

### 6.3 Column Schema

| # | Column | Type | Description & Observations |
|---|--------|------|----------------------------|
| 1 | `TransactionID` | string | Unique transaction key. Format `TX######`, zero-padded to 6 digits. Range: `TX000001`–`TX050000`. Rows are **shuffled** — IDs are sequential but dates are not. |
| 2 | `AccountID` | string | Customer account identifier. Format `AC#####`. Accounts **recur** across many transactions (e.g., `AC00359` appears in rows 19 and 34, `AC00270` appears in the middle sample with the same DeviceID and balance). |
| 3 | `TransactionAmount` | float | Transaction value in implied USD. Observed range: ~$7–$1,340+ in sample; ~1,774 transactions exceed $1,000; ~23,794 are ≥$200. |
| 4 | `TransactionDate` | string (datetime) | Date/time of transaction. **Two distinct formats are present** (see §7.1). Date range spans **2020–2025**. |
| 5 | `TransactionType` | categorical | `Debit` (38,747 / 77.5%) or `Credit` (11,253 / 22.5%). |
| 6 | `Location` | categorical | US city name. 30+ distinct cities observed: San Diego, Houston, Mesa, Raleigh, Atlanta, Oklahoma City, Seattle, Indianapolis, Detroit, Nashville, Albuquerque, Memphis, Fort Worth, Miami, Milwaukee, San Jose, Baltimore, Los Angeles, Las Vegas, San Francisco, Denver, Austin, Columbus, Sacramento, Kansas City, Omaha, Virginia Beach, Charlotte, Tucson, Louisville, Chicago, Philadelphia, San Antonio, Portland, Colorado Springs, New York, Dallas, and others. |
| 7 | `DeviceID` | string | Device identifier. Format `D######`. Devices **recur** across different accounts and locations (potential fraud signal). |
| 8 | `IP Address` | string | IPv4 address. **Column name contains a space** — requires `df['IP Address']` in pandas. IPs **recur** across accounts and geographically separate locations (see §7.2). |
| 9 | `MerchantID` | string | Merchant identifier. Format `M###`. Up to ~M099 observed — ~100 distinct merchants. |
| 10 | `Channel` | categorical | `ATM` (16,552 / 33.1%), `Branch` (17,278 / 34.6%), `Online` (16,170 / 32.3%). Near-uniform distribution. |
| 11 | `CustomerAge` | integer | Customer age in years. Range 18–79 observed. See §7.3 for demographic inconsistencies. |
| 12 | `CustomerOccupation` | categorical | Exactly 4 values: `Student` (13,059 / 26.1%), `Engineer` (12,491 / 25.0%), `Doctor` (12,578 / 25.2%), `Retired` (11,872 / 23.8%). Near-perfectly uniform — strongly suggests **synthetic assignment** (see §7.3). |
| 13 | `TransactionDuration` | integer | Duration in an unspecified unit. Observed range ~10–300 (likely seconds given the magnitude). |
| 14 | `LoginAttempts` | integer | Number of login attempts before the transaction. Predominantly `1`; elevated values (2, 3, 4, 5) appear in a minority of rows and are a likely fraud signal. |
| 15 | `AccountBalance` | float | Account balance at time of transaction (unclear if pre- or post-transaction). Range ~$137–$14,900+ observed. |

### 6.4 Date Coverage

| Approximate period | Count | Format present |
|---|---|---|
| 2020–2022 | ~21,815 | `M/D/YYYY` (date only) |
| 2023 (early rows) | ~3,107 | `M/D/YYYY H:MM` (with time) |
| 2023–2025 (later rows) | ~21,935 | `M/D/YYYY` (date only) |

Rows are **not in chronological order** — TransactionIDs are sequential but dates are randomized across the full 2020–2025 span, consistent with a shuffled or synthetically augmented dataset.

---

## 7. Data Quality & Anomalies

### 7.1 Inconsistent DateTime Format

The most significant data quality issue: `TransactionDate` uses **two different formats** across the file.

- **Rows 1–~25,000:** `M/D/YYYY H:MM` (e.g., `4/11/2023 16:29`) — includes time of day
- **Rows ~25,001–50,000:** `M/D/YYYY` (e.g., `5/19/2024`) — date only, no time

This almost certainly means the dataset was produced by **concatenating two separately-generated sources**, then running a cleaning pass. The cleaning did not normalize the date format. Consequences:
- Time-of-day features (e.g., "night-time transaction?", "business hours?") can only be extracted from the first half of the dataset.
- Any time-aware parsing (e.g., `pd.to_datetime(format=...)`) will fail unless a flexible parser is used.
- Any model trained on both halves will have implicit feature leakage or asymmetric missingness for time-derived features.

### 7.2 Recurring IPs and Devices Across Accounts

Multiple examples visible in the first 60 rows alone:

| IP / Device | Account 1 | Location 1 | Account 2 | Location 2 |
|---|---|---|---|---|
| IP `33.28.138.82` | AC00291 (TX000032) | Baltimore | AC00404 (TX000037) | Milwaukee |
| IP `92.214.76.157` | AC00069 (TX000008) | Indianapolis | AC00365 (TX000035) | San Francisco |
| IP `186.135.2.148` | AC00439 (TX000048) | Oklahoma City | AC00419 (TX000055) | Omaha |
| Device `D000297` | AC00390 (TX000030) | Detroit | AC00404 (TX000037) | Milwaukee |
| Device `D000671` | AC00317 (TX000017) | Austin | AC00041 (TX000026) | Houston |
| Device `D000235` | AC00019 (TX000003) | Mesa | AC00115 (TX000053) | Virginia Beach |

Shared IPs and devices across distinct accounts and geographically distant locations are a canonical fraud pattern. Whether these are genuine synthetic anomalies or incidental artifacts of how the data was generated is unclear, but they are highly relevant features for fraud model experimentation.

Additionally, some accounts appear with **identical DeviceID and AccountBalance** across different `TransactionID`s (e.g., `AC00270` + `D000466` + balance `3465.54` appears in both the early rows and the mid-file sample). This could reflect frozen/snapshot balance values or a synthetic data generation artifact.

### 7.3 Demographic Inconsistencies

The occupation distribution is nearly perfectly uniform (~25% each across Student, Engineer, Doctor, Retired), regardless of age. This is statistically implausible for a real-world population — strong evidence of **random or round-robin synthetic assignment**.

Visible example: `TX000021` — age 71 with occupation `Retired` is plausible, but the dataset contains `Student` records with ages in the 50s–70s and `Doctor` records with age 18, which would be demographically inconsistent in reality.

This limits the predictive value of `CustomerOccupation` as a feature if the sandbox is meant to simulate real-world fraud patterns.

### 7.4 LoginAttempts Distribution

The vast majority of rows show `LoginAttempts = 1`. Elevated values (2–5) appear in a small fraction of rows and are likely high-signal fraud indicators. The distribution is heavily right-skewed.

### 7.5 TransactionAmount Range

With ~1,774 transactions exceeding $1,000 (~3.5%) and ~23,794 exceeding $200 (~47.6%), high-value transactions are common. Large amounts combined with elevated login attempts, shared IPs, or unusual channel/location combinations are natural multi-feature fraud rule candidates.

### 7.6 No Fraud Label

There is **no target variable** in the dataset. No column named `IsFraud`, `FraudFlag`, `Label`, `Class`, or equivalent. This means:
- **Supervised classification is not directly possible** without an external label source or synthetic label generation.
- The dataset is best suited to **unsupervised or semi-supervised** approaches initially.
- Alternatively, domain rules (e.g., flagging `LoginAttempts > 3` OR shared IP across accounts) could be used as weak supervision to generate proxy labels.

---

## 8. Summary Table

| Property | Value |
|---|---|
| License | Apache 2.0 |
| Python version | 3.11.11 (Homebrew) |
| Venv name | `baTr/` |
| Jupyter kernel stack installed | Yes (ipykernel 7.2.0, IPython 9.10.1, jupyter_client 8.8.0) |
| Data science packages installed | **No** (pandas, numpy, sklearn, etc. absent) |
| Notebook status | Empty (1 blank code cell) |
| Dataset rows | 50,000 |
| Dataset columns | 15 |
| Date range | 2020–2025 |
| Date format | **Two formats** — with time (rows 1–~25k) and without (rows ~25k–50k) |
| TransactionType split | 77.5% Debit / 22.5% Credit |
| Channel split | ATM 33.1%, Branch 34.6%, Online 32.3% |
| Occupation split | ~25% each (Student / Engineer / Doctor / Retired) |
| High-value transactions (>$1,000) | ~1,774 (~3.5%) |
| Fraud label present | **No** |
| Notable data issues | Dual datetime formats, no label, synthetic occupation distribution, IP/device reuse across accounts, frozen balance values |

---

## 9. Potential Work Directions

### 9.1 Immediate Prerequisites
Before running any code:
1. Install data science packages into `baTr/`:
   ```
   source bankTransactions/baTr/bin/activate
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Add `baTr/` to `.gitignore` to avoid committing the venv.
3. Create a `requirements.txt` for reproducibility.

### 9.2 Exploratory Data Analysis (EDA)
- Distributions of `TransactionAmount`, `AccountBalance`, `TransactionDuration`, `LoginAttempts`
- Transaction volume over time (handle dual date formats first)
- Cross-tabulations: Channel × Type, Occupation × Age, Location × Amount
- IP and device reuse rates: how many IPs / devices are shared across 2+ accounts?

### 9.3 Feature Engineering
- **Time-based:** hour of day (first 25k only), day of week, month, year
- **Velocity:** transaction count per account per N hours/days
- **Aggregation:** mean/std amount per account, per device, per IP
- **Network/sharing flags:** `ip_shared_across_accounts`, `device_shared_across_accounts`
- **Risk flags:** `high_login_attempts` (> 1), `low_balance_high_amount` (amount > X% of balance)
- **Consistency:** location–IP mismatch heuristics (geolocation lookup would be needed)

### 9.4 Unsupervised Approaches (no label needed)
- **Isolation Forest** or **Local Outlier Factor** on numerical features
- **DBSCAN** or **K-Means** to identify unusual transaction clusters
- **Autoencoder** reconstruction error as anomaly score
- **Rule-based flagging** as a baseline: `LoginAttempts > 2` AND amount > threshold

### 9.5 Supervised Approaches (require labels)
- Synthetic label generation via domain rules (weak supervision)
- Joining with an external labeled dataset
- Models: Logistic Regression, Random Forest, XGBoost/LightGBM
- Imbalance handling: SMOTE, class weights, threshold tuning

### 9.6 Data Quality Remediation
- Normalize `TransactionDate` to a single format and extract time features where available
- Validate `CustomerAge` against `CustomerOccupation` (flag impossible combos)
- Audit repeated account+device+balance combos for synthetic artifacts
