Breast Cancer Classification with Logistic Regression
📌 Overview
This project uses the Breast Cancer Wisconsin Dataset to build a binary classification model that predicts whether a tumor is Malignant (M) or Benign (B) using Logistic Regression.

We go step-by-step through:

Data loading & cleaning

Train-test splitting & standardization

Model training

Model evaluation (confusion matrix, precision, recall, ROC-AUC)

Threshold tuning & sigmoid function explanation

📂 Dataset
Source: Kaggle – Breast Cancer Wisconsin (Diagnostic) dataset

File Used: breast_cancer.csv

Target Column:

diagnosis → M = Malignant (1), B = Benign (0)

Shape after cleaning:

Training set: (455, 30)

Test set: (114, 30)

⚙️ Steps
1. Load and Clean Data
python
Copy
Edit
import pandas as pd
df = pd.read_csv("breast_cancer.csv")
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
y = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop("diagnosis", axis=1)
2. Train-Test Split
python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
3. Standardize Features
python
Copy
Edit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
4. Train Logistic Regression Model
python
Copy
Edit
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
5. Model Evaluation
Confusion Matrix – visualizes TP, FP, TN, FN

Precision – proportion of predicted positives that are correct

Recall – proportion of actual positives correctly detected

ROC-AUC – measures the model’s ability to distinguish classes

📈 Results
Metric	Value
Precision	0.97
Recall	0.93
ROC-AUC	1.00

Threshold tuning results
Threshold	Precision	Recall
0.3	0.98	0.98
0.5 (default)	0.97	0.93
0.7	1.00	0.90

Insights:

Lower threshold (0.3) increases recall, catching more malignant cases.

Higher threshold (0.7) increases precision, reducing false positives but missing some true positives.

In medical contexts, higher recall is often preferred to minimize missed cancer detections.

🧮 Sigmoid Function in Logistic Regression
Logistic Regression outputs a probability between 0 and 1 using the sigmoid (logistic) function:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 
Where:

𝑧
=
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
⋯
+
𝑏
z=w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +⋯+b (linear combination of inputs)

Output = probability that the instance belongs to class 1 (Malignant in this case)

Key properties:

Always between 0 and 1

Output > 0.5 → predict class 1

Output < 0.5 → predict class 0

Smooth curve, useful for threshold tuning

📊 Visualizations
Confusion Matrix

ROC Curve

📌 How to Run
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
python breast_cancer_logreg.py
🧠 Key Takeaways
Logistic regression is fast, interpretable, and effective for binary classification.

Always scale features before training.

Threshold tuning is crucial when false negatives have serious consequences (e.g., cancer detection).

The sigmoid function is the mathematical heart of logistic regression, mapping any real number into a probability.

