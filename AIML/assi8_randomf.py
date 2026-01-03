# Ensemble Methods: Random Forest vs AdaBoost
# Classification, Evaluation, and Visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Step 1: Create Dataset
# -------------------------------
X, y = make_classification(
    n_samples=600,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 2: Random Forest Classifier
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)

cm_rf = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(cm_rf).plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

# -------------------------------
# Step 3: AdaBoost Classifier
# -------------------------------
ada = AdaBoostClassifier(
    n_estimators=150,
    random_state=42
)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

ada_acc = accuracy_score(y_test, ada_pred)
print("AdaBoost Accuracy:", ada_acc)

cm_ada = confusion_matrix(y_test, ada_pred)
ConfusionMatrixDisplay(cm_ada).plot()
plt.title("AdaBoost Confusion Matrix")
plt.show()
