import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, adjusted_rand_score, classification_report, accuracy_score, precision_score, recall_score
from scipy.stats import mode
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

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

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test_scaled)

gmm = GaussianMixture(n_components=len(np.unique(y_train_balanced)), random_state=42)
gmm.fit(X_train_pca)

y_test_cluster_gmm = gmm.predict(X_test_pca)

def map_clusters_to_labels(y_true, y_pred_clusters):
    labels = np.zeros_like(y_pred_clusters)
    for cluster in np.unique(y_pred_clusters):
        mask = y_pred_clusters == cluster
        labels[mask] = mode(y_true[mask], keepdims=False)[0]
    return labels

y_test_mapped = map_clusters_to_labels(y_test, y_test_cluster_gmm)

accuracy = accuracy_score(y_test, y_test_mapped)
precision = precision_score(y_test, y_test_mapped, average="weighted", zero_division=1)
recall = recall_score(y_test, y_test_mapped, average="weighted", zero_division=1)

print(f"精确率: {accuracy:.4f}")
print(f"预测率: {precision:.4f}")
print(f"回归率: {recall:.4f}")

report = classification_report(y_test, y_test_mapped, zero_division=1)
print("分类报告:\n", report)

cm = confusion_matrix(y_test, y_test_mapped)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("GMM混淆矩阵")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 聚类结果可视化（2D）
plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_cluster_gmm, cmap='viridis', alpha=0.7, edgecolor='k')
plt.title("2D聚类结果", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# 聚类结果可视化（3D）
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test_cluster_gmm, cmap='viridis', alpha=0.7, edgecolor='k')
ax.set_title("3D聚类结果")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.tight_layout()
plt.show()
