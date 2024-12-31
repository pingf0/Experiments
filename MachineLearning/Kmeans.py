import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv(r"E:\Git\train_data.csv")
test_data = pd.read_csv(r"E:\Git\test_data.csv")

X_train = train_data.drop(columns=["FaultType"])
y_train = train_data["FaultType"]
X_test = test_data.drop(columns=["FaultType"])
y_test = test_data["FaultType"]

common_features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[common_features]
X_test = X_test[common_features]

model = RandomForestClassifier(n_estimators=100, random_state=50)
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:50]
X_train = X_train.iloc[:, indices]
X_test = X_test.iloc[:, indices]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

inertia = []
for k in range(1, 21):
    kmeans_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_temp.fit(X_train_pca)
    inertia.append(kmeans_temp.inertia_)

plt.plot(range(1, 21), inertia, 'b-', marker='o')
plt.xlabel('数量K')
plt.ylabel('惯性值')
plt.title('肘部法则迭代')
plt.show()

best_k=4

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans.fit(X_train_pca)
y_pred_train = kmeans.predict(X_train_pca)
y_pred_test = kmeans.predict(X_test_pca)

cm = confusion_matrix(y_train, y_pred_train)
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
y_pred_test_aligned = np.array([mapping[label] for label in y_pred_test])

print("预测的标签分布：", np.unique(y_pred_test_aligned, return_counts=True))

accuracy = accuracy_score(y_test, y_pred_test_aligned)
precision = precision_score(y_test, y_pred_test_aligned, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred_test_aligned, average="weighted", zero_division=0)

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")

cm = confusion_matrix(y_test, y_pred_test_aligned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("混淆矩阵", fontsize=14)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.show()
