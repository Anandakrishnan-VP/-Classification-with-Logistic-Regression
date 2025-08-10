🩺 Breast Cancer Classification with Logistic Regression
This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset to build a binary classification model that predicts whether a tumor is Malignant (M) or Benign (B) using Logistic Regression.

📂 Dataset
Source: Kaggle – Breast Cancer Wisconsin (Diagnostic) Dataset

File Used: breast_cancer.csv

Target Column: diagnosis

M → Malignant (1)

B → Benign (0)

After cleaning:

Training set: (455, 30)

Test set: (114, 30)

⚙️ Project Workflow
1. Load & Clean Data
python
import pandas as pd

df = pd.read_csv("breast_cancer.csv")
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

y = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop("diagnosis", axis=1)
2. Train-Test Split
python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
3. Standardize Features
python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
4. Train Logistic Regression Model
python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
📊 Model Evaluation
Confusion Matrix: Visualizes TP, FP, TN, FN

Precision: Proportion of predicted positives that are correct

Recall: Proportion of actual positives correctly detected

ROC-AUC: Measures model’s ability to distinguish classes

Metric	Value
Precision	0.97
Recall	0.93
ROC-AUC	1.00
📉 Threshold Tuning Results
Threshold	Precision	Recall
0.3	0.98	0.98
0.5 (default)	0.97	0.93
0.7	1.00	0.90
Insights:

Lower threshold (0.3) increases recall, catching more malignant cases.

Higher threshold (0.7) increases precision, reducing false positives but missing some true positives.

For medical contexts, higher recall is preferred to reduce missed detections.

🧮 Sigmoid Function in Logistic Regression
The logistic regression outputs a probability between 0 and 1 using the sigmoid (logistic) function:

σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
 
where

z
=
w
1
x
1
+
w
2
x
2
+
⋯
+
b
z=w 
1
 x 
1
 +w 
2
 x 
2
 +⋯+b
Output is the probability of class 1 (Malignant).
Predict class 1 if output > 0.5, else class 0.
The function produces a smooth curve useful for threshold tuning.

📈 Visualizations
Confusion Matrix
![Confusion Matrix](images/confusion_curve.png)

![ROC Curve](images/roc_curve.png)

🚀 How to Run
bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the project
python breast_cancer_logreg.py
🧠 Key Takeaways
Logistic Regression is fast, interpretable, and effective for binary classification.
Always scale features prior to training.
Threshold tuning is crucial when false negatives have serious consequences (e.g., cancer detection).
The sigmoid function is central to logistic regression’s predictive power.


