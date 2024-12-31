import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

train_data = pd.read_csv(r"E:\Git\train_data.csv")
test_data = pd.read_csv(r"E:\Git\test_data.csv")

print("训练数据集：\n",train_data)
print("测试数据集：\n",test_data)

X_train = train_data.drop(columns=['FaultType'])
y_train = train_data['FaultType']
X_test = test_data.drop(columns=['FaultType'])
y_test = test_data['FaultType']

print("X_train columns:\n", X_train.columns)
print("X_test columns:\n", X_test.columns)

common_features = list(set(X_train.columns).intersection(set(X_test.columns)))
X_train = X_train[common_features]
X_test = X_test[common_features]

print(X_train)
print(y_train)
print(X_test)
print(y_test)

selector = RandomForestClassifier(n_estimators=100, random_state=50)
selector.fit(X_train, y_train)
feature_importances = selector.feature_importances_
important_indices = np.argsort(feature_importances)[::-1][:50]  #选择前50个重要特征
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

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)

    class_report = classification_report(y_test, y_pred, zero_division=1)

    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Classification Report:")
    print(class_report)

    return accuracy, recall, precision, f1, y_pred

logreg = LogisticRegression(max_iter=1000, C=10, random_state=42)
logreg_accuracy, logreg_recall, logreg_precision, logreg_f1, logreg_pred = evaluate_model(logreg, X_train_balanced, y_train_balanced, X_test_selected, y_test, "Logistic Regression")

svm = SVC(class_weight='balanced', random_state=42)
svm_accuracy, svm_recall, svm_precision, svm_f1, svm_pred = evaluate_model(svm, X_train_balanced, y_train_balanced, X_test_selected, y_test, "Support Vector Machine")

rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
rf_accuracy, rf_recall, rf_precision, rf_f1, rf_pred = evaluate_model(rf, X_train_balanced, y_train_balanced, X_test_selected, y_test, "Random Forest")

xgb = XGBClassifier()
xgb_accuracy, xgb_recall, xgb_precision, xgb_f1, xgb_pred = evaluate_model(xgb, X_train_balanced, y_train_balanced, X_test_selected, y_test, "XGBoost")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test_selected)

kmeans = KMeans(n_clusters=len(np.unique(y_train_balanced)), n_init=10, random_state=42)
kmeans.fit(X_train_pca)
y_test_cluster = kmeans.predict(X_test_pca)

kmeans_accuracy = adjusted_rand_score(y_test, y_test_cluster)
print(f"K-Means Clustering Adjusted Rand Index (ARI): {kmeans_accuracy:.4f}")

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(X_test_selected)

gmm = GaussianMixture(n_components=len(np.unique(y_train_balanced)), random_state=42)
gmm.fit(X_train_pca)

y_test_cluster_gmm = gmm.predict(X_test_pca)

gmm_ari = adjusted_rand_score(y_test, y_test_cluster_gmm)
print(f"GMM Clustering Adjusted Rand Index (ARI): {gmm_ari:.4f}")

accuracies = [logreg_accuracy, svm_accuracy, rf_accuracy, xgb_accuracy, gmm_ari]
models = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost', 'GMM (ARI)']

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, hue=models, dodge=False, palette='viridis', legend=False)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.title('Model Accuracy Comparison (Including GMM ARI)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy / ARI', fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, logreg_pred, "Logistic Regression")
plot_confusion_matrix(y_test, svm_pred, "Support Vector Machine")
plot_confusion_matrix(y_test, rf_pred, "Random Forest")
plot_confusion_matrix(y_test, xgb_pred, "XGBoost")

models_with_predictions = [
    ("Logistic Regression", logreg),
    ("Support Vector Machine", svm),
    ("Random Forest", rf),
    ("XGBoost", xgb),
]

plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_cluster_gmm, cmap='viridis', alpha=0.7, edgecolor='k')
plt.title("GMM Clustering - Test Data", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 8))
for i, (name, model) in enumerate(models_with_predictions, 1):
    y_train_pred = model.predict(X_train_balanced)
    y_test_pred = model.predict(X_test_selected)

    plt.subplot(len(models_with_predictions), 2, 2 * (i - 1) + 1)
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pred, cmap="viridis", alpha=0.6, edgecolor='k', linewidth=0.5)
    plt.title(f"{name} - Training Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.subplot(len(models_with_predictions), 2, 2 * (i - 1) + 2)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pred, cmap="coolwarm", alpha=0.6, edgecolor='k', linewidth=0.5)
    plt.title(f"{name} - Test Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 8))
for i, (name, model) in enumerate(models_with_predictions, 1):
    y_train_pred = model.predict(X_train_balanced)
    y_test_pred = model.predict(X_test_selected)

    ax_train = fig.add_subplot(len(models_with_predictions), 2, 2 * (i - 1) + 1, projection='3d')
    ax_test = fig.add_subplot(len(models_with_predictions), 2, 2 * (i - 1) + 2, projection='3d')

    ax_train.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train_pred, cmap="viridis", alpha=0.6, edgecolor='k', linewidth=0.5)
    ax_train.set_title(f"{name} - Training Data")
    ax_train.set_xlabel("Principal Component 1")
    ax_train.set_ylabel("Principal Component 2")
    ax_train.set_zlabel("Principal Component 3")

    ax_test.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test_pred, cmap="coolwarm", alpha=0.6, edgecolor='k', linewidth=0.5)
    ax_test.set_title(f"{name} - Test Data")
    ax_test.set_xlabel("Principal Component 1")
    ax_test.set_ylabel("Principal Component 2")
    ax_test.set_zlabel("Principal Component 3")

plt.tight_layout()
plt.show()
