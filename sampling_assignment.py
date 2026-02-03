# ==========================================
# Sampling Assignment - FINAL VERSION
# ==========================================

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# ------------------------------------------
np.random.seed(43)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/Creditcard_data.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nOriginal Distribution:\n", y.value_counts())

# ==========================================
# 2. BALANCE DATASET (UNDERSAMPLING)
# ==========================================

majority = df[df.Class == 0]
minority = df[df.Class == 1]

majority_down = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

df_bal = pd.concat([majority_down, minority])

X_bal = df_bal.drop("Class", axis=1)
y_bal = df_bal["Class"]

print("\nBalanced Distribution:\n", y_bal.value_counts())

# ==========================================
# 3. SAMPLING TECHNIQUES
# ==========================================

# ---------- Random Sampling ----------
def simple_random_sampling(X, y, sample_size=500):
    sample_size = min(sample_size, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    return X.iloc[idx], y.iloc[idx]

# ---------- Systematic Sampling ----------
def systematic_sampling(X, y, step=3):
    # Shuffle indices first
    perm = np.random.permutation(len(X))

    # Take every k-th element
    idx = perm[::step]

    return X.iloc[idx], y.iloc[idx]


# ---------- Stratified Sampling ----------
def stratified_sampling(X, y, frac=0.7):
    Xs, _, ys, _ = train_test_split(
        X, y,
        train_size=frac,
        stratify=y,
        random_state=42
    )
    return Xs, ys

# ---------- Bootstrap Sampling ----------
def bootstrap_sampling(X, y, sample_size=500):
    sample_size = min(sample_size, len(X))
    Xs, ys = resample(
        X, y,
        n_samples=sample_size,
        replace=True,
        random_state=42
    )
    return Xs, ys

# ---------- Cross Validation Sampling ----------
def cv_sampling(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, _ in kf.split(X):
        return X.iloc[train_idx], y.iloc[train_idx]

samplers = {
    "Random": simple_random_sampling,
    "Systematic": systematic_sampling,
    "Stratified": stratified_sampling,
    "Bootstrap": bootstrap_sampling,
    "CrossVal": cv_sampling
}

# ==========================================
# 4. MODELS
# ==========================================

models = {
    "M1": LogisticRegression(max_iter=500),
    "M2": RandomForestClassifier(random_state=42),
    "M3": KNeighborsClassifier(n_neighbors=3),
    "M4": SVC(random_state=42),
    "M5": DecisionTreeClassifier(random_state=42)
}

# ==========================================
# 5. TRAIN FUNCTION
# ==========================================

def train_model(model, X, y):

    # Safety check for very small class
    if y.value_counts().min() < 2:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    if isinstance(model, KNeighborsClassifier):
        model.set_params(n_neighbors=min(3, len(Xtr)))

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    return accuracy_score(yte, pred)

# ==========================================
# 6. RUN EXPERIMENT
# ==========================================

results = {}

for sname, sampler in samplers.items():

    if sname in ["Random", "Bootstrap"]:
        Xs, ys = sampler(X_bal, y_bal, sample_size=500)
    else:
        Xs, ys = sampler(X_bal, y_bal)

    # Final safety
    if ys.nunique() < 2:
        Xs, ys = X_bal, y_bal

    results[sname] = {}

    for mname, model in models.items():
        acc = train_model(model, Xs, ys)
        results[sname][mname] = round(acc * 100, 2)

# ==========================================
# 7. RESULTS TABLE
# ==========================================

result_df = pd.DataFrame(results)

print("\n================ ACCURACY TABLE ================\n")
print(result_df)

# ==========================================
# 8. BEST SAMPLING PER MODEL
# ==========================================

best_sampling = result_df.idxmax(axis=1)

print("\n=========== BEST SAMPLING PER MODEL ===========\n")
print(best_sampling)

# ==========================================
# 9. SAVE RESULTS
# ==========================================

result_df.to_csv("results/accuracy_table.csv")
print("\nSaved to results/accuracy_table.csv")
