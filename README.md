# 📊 Sampling Assignment — Credit Card Fraud Detection

This repository contains a Python implementation to explore how different **sampling techniques** affect the performance of various **machine learning classifiers** on an imbalanced credit card fraud dataset.

Imbalanced data is common in fraud detection problems. Without proper handling, models tend to predict only the majority (legitimate) class, failing to detect the rare but important fraudulent transactions. This project balances the data and compares multiple sampling strategies and models.

---

## 🔍 Repository Contents


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
- Labels:

| Value | Meaning                    |
|-------|---------------------------|
| 0     | Legitimate transaction    |
| 1     | Fraudulent transaction    |

---

## 🧪 Sampling Techniques Used

The following sampling techniques are tested on the balanced dataset:

| Technique      | Description                                                        |
|---------------|--------------------------------------------------------------------|
| **Random**     | Random sampling without replacement                                |
| **Systematic** | Picks every *k-th* row after shuffling                             |
| **Stratified** | Preserves class ratios when splitting                             |
| **Bootstrap**  | Random sampling with replacement                                   |
| **CrossVal**   | Uses training portion of a K-Fold split                            |

---

## 🧠 Machine Learning Models

The following classification models are trained on each sampled dataset:

| Model Code | Model Type                 |
|------------|---------------------------|
| M1         | Logistic Regression       |
| M2         | Random Forest Classifier |
| M3         | K-Nearest Neighbors       |
| M4         | Support Vector Machine    |
| M5         | Decision Tree Classifier  |

---

## 🚀 How It Works (Step-by-Step)

### 1️⃣ Load and Balance Dataset

The script loads the CSV dataset, then uses undersampling to balance the number of fraud and non-fraud records.

```python
df_bal = pd.concat([majority_down, minority])
## 2️⃣ Apply Sampling Methods

Each sampling method generates a new subset of the balanced dataset for model training.

**Example — Random Sampling**

```python
idx = np.random.choice(len(X), sample_size, replace=False)
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

Results are saved at:

results/accuracy_table.csv