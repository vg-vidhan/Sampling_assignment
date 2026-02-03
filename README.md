# 📊 Sampling Assignment — Credit Card Fraud Detection

This repository contains a Python implementation to explore how different **sampling techniques** affect the performance of various **machine learning classifiers** on an imbalanced credit card fraud dataset.

Imbalanced data is common in fraud detection problems. Without proper handling, models tend to predict only the majority (legitimate) class, failing to detect the rare but important fraudulent transactions. This project balances the data and compares multiple sampling strategies and models.

---

## 🔍 Repository Contents

Sampling_assignment/
│
├── data/
│ └── Creditcard_data.csv
├── sampling_assignment.py
├── results/
│ └── accuracy_table.csv
├── README.md


---

## 🧠 Project Goals

1. Handle class imbalance via undersampling  
2. Implement multiple sampling techniques  
3. Train five different classifiers  
4. Compare model performance across sampling methods  
5. Identify which sampling strategy works best per model  

---

## 📊 Dataset

- File path: `data/Creditcard_data.csv`  
- Target Column: `Class`

| Value | Meaning |
|------:|--------|
| 0 | Legitimate transaction |
| 1 | Fraudulent transaction |

---

## 🧪 Sampling Techniques Used

| Technique | Description |
|---------|-------------|
| Random | Random sampling without replacement |
| Systematic | Picks every k-th row after shuffling |
| Stratified | Preserves class ratios |
| Bootstrap | Sampling with replacement |
| CrossVal | Uses training fold of K-Fold |

---

## 🧠 Machine Learning Models

| Code | Model |
|-----|------|
| M1 | Logistic Regression |
| M2 | Random Forest |
| M3 | K-Nearest Neighbors |
| M4 | Support Vector Machine |
| M5 | Decision Tree |

---

## 🚀 How It Works (Step-by-Step)

---

1️⃣ Load and Balance Dataset

The script loads the CSV dataset, then uses undersampling to balance the number of fraud and non-fraud records.

df_bal = pd.concat([majority_down, minority])

2️⃣ Apply Sampling Methods

Each sampling method generates a new subset of the balanced dataset for model training.

Example — Random Sampling

idx = np.random.choice(len(X), sample_size, replace=False)
Xs = X.iloc[idx]
ys = y.iloc[idx]

3️⃣ Train Models and Evaluate

Each sampled dataset is split into training and test sets (80/20).

Models are trained on the training set and accuracy is recorded on the test set.

model.fit(Xtr, ytr)
accuracy_score(yte, pred)

🧾 Results

The script prints and saves a CSV file showing accuracy scores for each model on each sampling method.

           Random  Systematic  Stratified  Bootstrap  CrossVal
M1         85.23        65.10        88.45       81.65      87.12
M2         90.12        70.98        92.45       85.34      91.80
...


The best sampling method is also determined for each model based on highest accuracy.

📁 Saved Output
results/accuracy_table.csv