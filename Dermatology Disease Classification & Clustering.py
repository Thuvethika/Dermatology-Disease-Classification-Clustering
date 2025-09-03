# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import time

# ---------------------------------------------
#               LOAD & CLEAN DATA
# ---------------------------------------------

# Load dataset using tab separator
df = pd.read_csv("dermatology.csv", sep='\t')

# Clean column names for consistency
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")  # Replace spaces and hyphens
df.columns = df.columns.str.replace(".", "", regex=False)  # Remove periods

# Fix specific known typos
df = df.rename(columns={
    "Family_Hostory": "Family_History",
    "Follicular1": "Follicular_1"
})
# Check and display missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values[missing_values > 0])

# Drop rows containing missing values
df.dropna(inplace=True)

# Convert Disease column to integer
df['Disease'] = df['Disease'].astype(int)

# Display unique disease classes
print("Unique diseases:", df["Disease"].unique())

# Preview the first few rows
print(df.head())

# Convert Age to numeric and drop rows where conversion failed
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.dropna(subset=['Age'], inplace=True)

# ---------------------------------------------
#       MODEL 1: Gradient Descent (Age only)
# ---------------------------------------------


print("\n" + "="*80)
print("MODEL 1: AGE-BASED DISEASE PREDICTION USING GRADIENT DESCENT")
print("="*80)

# Extract Age and Disease
X_age = df[['Age']].values
y = df['Disease'].values

# Normalize Age (Min-Max)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_age)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Manual Gradient Descent
# Code Adapted from Crypto1, 2020
def gradient_descent(X, y, learning_rate=0.1, epochs=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)
    global mse_history
    mse_history = []
    for epoch in range(epochs):
        pred = X_b.dot(theta)
        gradients = 2/m * X_b.T.dot(pred - y.reshape(-1, 1))
        theta -= learning_rate * gradients
        mse = mean_squared_error(y, pred)
        mse_history.append(mse)

    return theta

# Predict function
def predict(X, theta):
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(theta)

# Convert regression output to class labels
def regression_to_class(y_pred, n_classes=6):
    return np.clip(np.round(y_pred), 1, n_classes).astype(int)

# Train
start_time = time.time()
theta = gradient_descent(X_train, y_train, learning_rate=0.1, epochs=1000)
train_time = time.time() - start_time

# Predict
y_pred_reg = predict(X_test, theta)
y_pred_cls = regression_to_class(y_pred_reg)

# Evaluate
mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
accuracy = accuracy_score(y_test, y_pred_cls)

print(f"\nModel Parameters:")
print(f"  Intercept (Theta0): {theta[0][0]:.4f}")
print(f"  Slope     (Theta1): {theta[1][0]:.4f}")
print(f"Training Time: {train_time:.4f} seconds")

print("\nRegression Metrics:")
print(f"  MSE : {mse:.4f}")
print(f"  MAE : {mae:.4f}")
print(f"  R  2: {r2:.4f}")

print("\nClassification Metrics (from rounded regression):")
print(f"  Accuracy: {accuracy:.4f}")

# ---------------------------------------------
# Plot: MSE Curve, Regression Fit, Confusion Matrix, Predicted vs True
# ---------------------------------------------

plt.figure(figsize=(14, 10))

# 1. MSE Loss Curve
plt.subplot(2, 2, 1)
plt.plot(mse_history, color='teal')
plt.title("Gradient Descent Convergence")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.grid(True)

# 2. Regression Plot
# Code adapted from Seaborn.scatterplot — Seaborn 0.11.1
plt.subplot(2, 2, 2)
age_test = scaler.inverse_transform(X_test).flatten()
plt.scatter(age_test, y_test, alpha=0.6, label='Actual')
age_line = np.linspace(min(age_test), max(age_test), 100).reshape(-1,1)
y_line = predict(scaler.transform(age_line), theta)
plt.plot(age_line, y_line, 'r-', label='Predicted Line')
plt.title("Regression Fit (Age vs Disease)")
plt.xlabel("Age")
plt.ylabel("Predicted Disease")
plt.legend()

# 3. Confusion Matrix
# Code adapted from Waskom, 2024
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(1, 7), yticklabels=range(1, 7))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

# 4. Predicted vs True Scatter Plot
# Code adapted from Seaborn.scatterplot — Seaborn 0.11.1
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_cls, alpha=0.6, c='purple')
plt.xlabel("True Disease")
plt.ylabel("Predicted Disease")
plt.title("Predicted vs True (Rounded Labels)")
plt.grid(True)

plt.tight_layout()
plt.savefig("model1_all_plots_combined.png", dpi=300)
plt.show()

# ---------------------------------------------
#        MODEL 2: RANDOM FOREST CLASSIFIER
# ---------------------------------------------

print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("="*80)

# Code adapted from GeeksforGeeks, 2024
# 1. Data Preparation
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# 2. Define Features and Labels
features = df.drop(columns=["Disease"])
labels = df["Disease"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42,
)

# 4. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predictions
y_pred = rf_model.predict(X_test)

# 6. Evaluation Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=[
        'Psoriasis', 'Seboreic Dermatitis', 'Lichen Planus', 
        'Pityriasis Rosea', 'Cronic Dermatitis', 'Pityriasis Rubra Pilaris'
    ]))

# 7. Confusion Matrix Plot
# Code adapted from Waskom, 2024
plt.figure(figsize=(6, 5))
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(labels.unique()), yticklabels=sorted(labels.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Model 2 - Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("model2_confusion_matrix.png", dpi=300)
plt.show()

# 8. Feature Importance Plot
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Model 2 - Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("model2_feature_importance.png", dpi=300)
plt.show()

# ---------------------------------------------
#        MODEL 3: K-NEAREST NEIGHBORS
# ---------------------------------------------

print("\n" + "="*80)
print("MODEL 3: K-NEAREST NEIGHBORS")
print("="*80)

# Code adapted from GeeksforGeeks, 2017
# 1. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 2. Train-Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. Evaluate kNN for multiple k values
k_values = [3, 4, 5,]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"\nAccuracy for k={k}: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        'Psoriasis', 'Seboreic Dermatitis', 'Lichen Planus', 
        'Pityriasis Rosea', 'Cronic Dermatitis', 'Pityriasis Rubra Pilaris'
    ]))
    
    # Confusion matrix plot
    # Code adapted from Waskom, 2024
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Purples",
                xticklabels=sorted(labels.unique()), yticklabels=sorted(labels.unique()))
    plt.title(f"Confusion Matrix - kNN (k={k})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"model3_confusion_matrix_k{k}.png", dpi=300)
    plt.show()

# 4. Accuracy vs k plot
accuracies = []
k_range = range(1, 20)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o')
plt.title("Accuracy vs k (kNN)")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("model3_accuracy_vs_k.png", dpi=300)
plt.show()

# ---------------------------------------------
#   VISUALIZATION WITH PCA AND T-SNE
# ---------------------------------------------

# Code adapted from (PCA vs T-SNE, n.d.)
# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
plt.title("PCA: 2D Projection of Dermatology Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Disease")
plt.grid(True)
plt.tight_layout()
plt.savefig("model3_pca_projection.png", dpi=300)
plt.show()

# t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="tab10", s=60, alpha=0.8)
plt.title("t-SNE: 2D Projection of Dermatology Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Disease")
plt.grid(True)
plt.tight_layout()
plt.savefig("model3_tsne_projection.png", dpi=300)
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)  # features from your dataframe

true_labels = labels.values  # true disease labels (for evaluation)
k = len(np.unique(true_labels))  # number of disease classes

# ---------------------------------------------
# Model 4: KMeans Clustering
# ---------------------------------------------
print("\n" + "="*80)
print("MODEL 4&5: KMeans Clustering and Agglomerative Clustering")
print("="*80)

# Code adapted from (Kavlakoglu & Winland, 2024)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
nmi_kmeans = normalized_mutual_info_score(true_labels, kmeans_labels)
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)

# ---------------------------------------------
# Model 5: Agglomerative Clustering
# ---------------------------------------------

# Code adapted from Sklearn.cluster.AgglomerativeClustering Scikit learn
agglo = AgglomerativeClustering(n_clusters=k)
agglo_labels = agglo.fit_predict(X_scaled)

ari_agglo = adjusted_rand_score(true_labels, agglo_labels)
nmi_agglo = normalized_mutual_info_score(true_labels, agglo_labels)
silhouette_agglo = silhouette_score(X_scaled, agglo_labels)


# Purity Score Function
def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

purity_kmeans = purity_score(true_labels, kmeans_labels)
purity_agglo = purity_score(true_labels, agglo_labels)

# Print Evaluation Metrics
print("Model 4 - KMeans Clustering")
print(f"Adjusted Rand Index (ARI): {ari_kmeans:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_kmeans:.4f}")
print(f"Purity Score: {purity_kmeans:.4f}")
print(f"Silhouette Score: {silhouette_kmeans:.4f}")
print(f"Cluster Sizes:\n{pd.Series(kmeans_labels).value_counts()}\n")

print("Model 5 - Agglomerative Clustering")
print(f"Adjusted Rand Index (ARI): {ari_agglo:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_agglo:.4f}")
print(f"Purity Score: {purity_agglo:.4f}")
print(f"Silhouette Score: {silhouette_agglo:.4f}")
print(f"Cluster Sizes:\n{pd.Series(agglo_labels).value_counts()}\n")


# PCA Projection for Clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "True_Label": true_labels.astype(str),
    "KMeans_Label": kmeans_labels.astype(str),
    "Agglo_Label": agglo_labels.astype(str)
})

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="KMeans_Label", palette="Set2")
plt.title("KMeans Clustering (PCA)")

plt.subplot(1, 3, 2)
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Agglo_Label", palette="Set1")
plt.title("Agglomerative Clustering (PCA)")

plt.subplot(1, 3, 3)
sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="True_Label", palette="tab10")
plt.title("True Labels (PCA)")

plt.tight_layout()
plt.show()

# t-SNE Projection
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=kmeans_labels, palette="Set2")
plt.title("KMeans Clustering (t-SNE)")

plt.subplot(1, 3, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=agglo_labels, palette="Set1")
plt.title("Agglomerative Clustering (t-SNE)")

plt.subplot(1, 3, 3)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=true_labels, palette="tab10")
plt.title("True Labels (t-SNE)")

plt.tight_layout()
plt.show()

# Dendrogram for Agglomerative Clustering
# Code adapted from Plot Hierarchical Clustering Dendrogram, Scikit learn
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Final Model Comparison (Summary Plot)
# ---------------------------------------------
summary_scores = {
    "Model 1 (GD Accuracy)": accuracy,
    "Model 2 (RF Accuracy)": acc,
    "Model 3 (kNN Best)": max(accuracies),
    "Model 4 (KMeans ARI)": ari_kmeans,
    "Model 5 (Agglo ARI)": ari_agglo
}

print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)
print(f"{'Model':<35} {'Metric':<15} {'Value':<10}")
print("-" * 60)
print(f"{'Model 1: Gradient Descent':<35} {'Accuracy':<15} {summary_scores['Model 1 (GD Accuracy)']:.4f}")
print(f"{'Model 2: Random Forest':<35} {'Accuracy':<15} {summary_scores['Model 2 (RF Accuracy)']:.4f}")
print(f"{'Model 3: kNN':<35} {'Accuracy':<15} {summary_scores['Model 3 (kNN Best)']:.4f}")
print(f"{'Model 4: Agglomerative Clustering':<35} {'ARI':<15} {summary_scores['Model 4 (KMeans ARI)']:.4f}")
print(f"{'Model 5: KMeans Clustering':<35} {'ARI':<15} {summary_scores['Model 5 (Agglo ARI)']:.4f}")

models = list(summary_scores.keys())
scores = list(summary_scores.values())

# Custom colors (5 different)
colors = ['skyblue', 'lightgreen', 'lightsalmon', 'plum', 'gold']

# Create horizontal bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(models, scores, color=colors)

# Add value labels to bars
for bar, value in zip(bars, scores):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{value:.4f}", va='center', fontsize=10, fontweight='bold')

# Labels & styling
plt.xlabel("Score (Accuracy / ARI)")
plt.title("Final Performance Comparison Across Models")
plt.xlim(0, 1.1)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("final_model_comparison.png", dpi=300)
plt.show()