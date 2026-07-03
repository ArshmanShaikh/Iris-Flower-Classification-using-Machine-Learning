# ============================================
# PROJECT 2: DATA CLASSIFICATION USING AI
# Iris Classification using Machine Learning
# ============================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# STEP 1: Load Dataset
# ==============================
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

print("First Five Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==============================
# STEP 2: Data Exploration
# ==============================
print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ==============================
# STEP 3: Visualization
# ==============================

# Pair Plot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# STEP 4: Split Features & Target
# ==============================
X = df.iloc[:, :-1]
y = df["Species"]

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# STEP 5: Train Model
# ==============================
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==============================
# STEP 6: Prediction
# ==============================
y_pred = model.predict(X_test)

# ==============================
# STEP 7: Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ==============================
# STEP 8: Confusion Matrix Plot
# ==============================
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# STEP 9: Feature Importance
# ==============================
importance = pd.Series(
    model.feature_importances_,
    index=iris.feature_names
)

importance.sort_values().plot(
    kind="barh",
    figsize=(7, 4)
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

print("\nProject Completed Successfully!")