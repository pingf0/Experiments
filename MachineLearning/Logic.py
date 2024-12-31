import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, label_binarize

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv(r"E:\Git\train_data.csv")
test_data = pd.read_csv(r"E:\Git\test_data.csv")

X_train = train_data.drop(columns=["FaultType"])
y_train = train_data["FaultType"]
X_test = test_data.drop(columns=["FaultType"])
y_test = test_data["FaultType"]

common_features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[common_features]
X_test = X_test[common_features]

selector = RandomForestClassifier(n_estimators=100, random_state=50)
selector.fit(X_train, y_train)
feature_importances = selector.feature_importances_
important_indices = np.argsort(feature_importances)[::-1][:50]  # 选择前50个重要特征
X_train_selected = X_train.iloc[:, important_indices]
X_test_selected = X_test.iloc[:, important_indices]

scaler = StandardScaler()
X_train_selected = scaler.fit_transform(X_train_selected)
X_test_selected = scaler.transform(X_test_selected)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

unique, counts = np.unique(y_train_balanced, return_counts=True)
print("Balanced training class distribution:")
print(dict(zip(unique, counts)))

logreg = LogisticRegression(max_iter=1000, C=10, random_state=42, class_weight="balanced")
logreg.fit(X_train_balanced, y_train_balanced)
y_pred = logreg.predict(X_test_selected)
y_pred_proba = logreg.predict_proba(X_test_selected)

unique, counts = np.unique(y_pred, return_counts=True)
print("Predicted labels distribution:", dict(zip(unique, counts)))

precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print(f"预测率: {precision:.4f}")
print(f"回归率: {recall:.4f}")

print("分类报告:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("逻辑回归混淆矩阵", fontsize=14)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.show()

y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))  # 将测试集标签二值化
precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
average_precision_micro = average_precision_score(y_test_binarized, y_pred_proba, average="micro")

plt.figure(figsize=(8, 6))
plt.plot(recall_micro, precision_micro, label=f"平均精确度 = {average_precision_micro:.4f}")
plt.title("精确度-召回率曲线", fontsize=14)
plt.xlabel("召回率", fontsize=12)
plt.ylabel("精确度", fontsize=12)
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(y_test, bins=np.arange(len(np.unique(y_test)) + 1) - 0.5, alpha=0.7, color='blue', label='真实标签')
plt.title("真实标签分布", fontsize=14)
plt.xlabel("标签", fontsize=12)
plt.ylabel("数量", fontsize=12)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_pred, bins=np.arange(len(np.unique(y_test)) + 1) - 0.5, alpha=0.7, color='green', label='预测标签')
plt.title("预测标签分布", fontsize=14)
plt.xlabel("标签", fontsize=12)
plt.ylabel("数量", fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

