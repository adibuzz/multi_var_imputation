from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Create an imbalanced dataset
X, y = make_classification(
    n_samples=4000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # 90% of class 0, 10% of class 1
    random_state=42
)

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Step 3: Create data subsets for each model
subset_size = len(X_train) // 3
X1, y1 = X_train[:subset_size], y_train[:subset_size]
X2, y2 = X_train[subset_size:2*subset_size], y_train[subset_size:2*subset_size]
X3, y3 = X_train[2*subset_size:], y_train[2*subset_size:]

# Step 4: Define models
model1 = LogisticRegression(max_iter=1000, class_weight='balanced')
model2 = DecisionTreeClassifier(max_depth=6, class_weight='balanced')
model3 = SVC(probability=True, class_weight='balanced')

# Step 5: Train individual models
model1.fit(X1, y1)
model2.fit(X2, y2)
model3.fit(X3, y3)

# Step 6: Ensemble model (Voting Classifier)
ensemble = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('svc', model3)],
    voting='soft'  # use predicted probabilities
)

p = ensemble.fit(X_train, y_train)
print(p)
# # Step 7: Evaluate results
print("Model 1 accuracy:", model1.score(X_test, y_test))
print("Model 2 accuracy:", model2.score(X_test, y_test))
print("Model 3 accuracy:", model3.score(X_test, y_test))
print("Ensemble accuracy:", ensemble.score(X_test, y_test))

# Step 8: More detailed evaluation
y_pred = ensemble.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
