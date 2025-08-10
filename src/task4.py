import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#data preprocessing 
df=pd.read_csv("breast_cancer.csv")
print(df.isnull().sum())
df=df.drop(columns=["Unnamed: 32", "id"])
x=df.drop(columns=['diagnosis'])
y=df['diagnosis'].map({'M':1,'B':0})

x_train, x_test, y_train, y_test= train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)

#Fit a Linear Regression model
model= LogisticRegression()
model.fit(x_train, y_train)

#Evaluate with confusion matrix, precision, recall, ROC-AUC.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(x_test)
#Confusion matrix
cm= confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Precision & Recall
precision=precision_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:2f}")

#ROC-AUC
y_prob = model.predict_proba(x_test)[:, 1]  # Probabilities for class 1
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal = random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Tune classification threshold to observe changes in precision and recall
import numpy as np
from sklearn.metrics import precision_score, recall_score
#Probabilties for class 1
y_prob = model.predict_proba(x_test)[:, 1]
#different thresholds
thresholds = [0.3, 0.5, 0.7]
for t in thresholds:
    y_pred_custom = (y_prob >= t).astype(int)
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    print(f"Threshold: {t}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print()