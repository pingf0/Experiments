import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv(r"E:\Git\train_data.csv")
test_data = pd.read_csv(r"E:\Git\test_data.csv")

X_train = train_data.drop(columns=['FaultType'])
y_train = train_data['FaultType']
X_test = test_data.drop(columns=['FaultType'])
y_test = test_data['FaultType']

common_features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[common_features]
X_test = X_test[common_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

svm = SVC(class_weight='balanced', random_state=42)
svm.fit(X_train_balanced, y_train_balanced)

y_pred = svm.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)

print(f"精确率: {accuracy:.4f}")
print(f"预测率: {precision:.4f}")
print(f"回归率: {recall:.4f}")
print("分类报告:")
print(classification_report(y_test, y_pred, zero_division=1))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("SVM混淆矩阵")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
