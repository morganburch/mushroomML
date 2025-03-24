import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
"""note LLM used in decelopment of this code"""

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
#load data, encode, split
data = np.genfromtxt('mushrooms.csv', delimiter=',', dtype=str)

X = data[1:, 1:]  
y = data[1:, 0]  

label_encoders = {}
for i in range(X.shape[1]):
    le = {label: idx for idx, label in enumerate(np.unique(X[:, i]))}
    X[:, i] = [le[label] for label in X[:, i]]
    label_encoders[i] = le

le_y = {label: idx for idx, label in enumerate(np.unique(y))}
y = [le_y[label] for label in y]

#split data
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


#testing k values
k_values = range(100, 105)
accuracies = []

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train.astype(float), y_train)
    predictions = knn.predict(X_test.astype(float))
    accuracy = np.mean(predictions == y_test)
    accuracies.append(accuracy)

for k, accuracy in zip(k_values, accuracies):
    print(f'Accuracy for k={k}: {accuracy:.3f}')

best_k = k_values[np.argmax(accuracies)]
print(f'Best k value: {best_k}')
conf_matrix = confusion_matrix(y_test, knn.predict(X_test.astype(float)))
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Edible (0)', 'Actual Poisonous (1)'],
                              columns=['Predicted Edible (0)', 'Predicted Poisonous (1)'])

print("Confusion Matrix:")
print(conf_matrix_df)
