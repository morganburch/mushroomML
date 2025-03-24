import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
"""note LLM used in decelopment of this code
   Also note it may take a minute for this program to run/complete due to grid search 
"""

data = np.genfromtxt('mushrooms.csv', delimiter=',', dtype=str)

# Extract features and target variable
X = data[1:, 1:]  
y = data[1:, 0]  

label_encoders = []
for i in range(X.shape[1]):
    le = LabelEncoder()
    X[:, i] = le.fit_transform(X[:, i])
    label_encoders.append(le)

le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Splitting data into 70% training and 30% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Standardizing the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],  # Ensure compatibility with chosen penalty
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced']
}

log_reg = LogisticRegression()

#grid search
grid_search = GridSearchCV(log_reg, param_grid, cv=5)
grid_search.fit(X_train_std, y_train)
print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_std)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Edible (0)', 'Actual Poisonous (1)'],
                              columns=['Predicted Edible (0)', 'Predicted Poisonous (1)'])

print(f"Accuracy: {accuracy:.3f}")
print(conf_matrix_df)
print(class_report)
