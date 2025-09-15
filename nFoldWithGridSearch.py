import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    """Preprocess the data by encoding categorical features and splitting into train/test sets."""
    # Extract features and target variable
    X = data[1:, 1:]
    y = data[1:, 0]

    # Encode categorical data
    label_encoders = []
    for i in range(X.shape[1]):
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def n_fold_cross_validation(X, y, model, n=5):
    """Perform n-fold cross-validation."""
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split the data into n folds
    fold_size = X.shape[0] // n
    folds_X = [X[i*fold_size:(i+1)*fold_size] for i in range(n)]
    folds_y = [y[i*fold_size:(i+1)*fold_size] for i in range(n)]

    accuracies = []
    confusion_matrices = []

    # Perform n-fold cross-validation
    for i in range(n):
        # Create training and validation sets
        X_train = np.vstack([folds_X[j] for j in range(n) if j != i])
        y_train = np.hstack([folds_y[j] for j in range(n) if j != i])
        X_val = folds_X[i]
        y_val = folds_y[i]

        # Standardizing the features
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_val_std = sc.transform(X_val)

        # Train the model
        model.fit(X_train_std, y_train)

        # Evaluate the model
        y_pred = model.predict(X_val_std)

        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        accuracies.append(acc)
        confusion_matrices.append(cm)

    return accuracies, confusion_matrices

def custom_grid_search(X, y, param_grid, n=5):
    """Perform custom grid search with cross-validation to find the best parameters."""
    best_score = 0
    best_params = None
    best_confusion_matrix = None
    best_accuracies = None

    for C in param_grid['C']:
        for penalty in param_grid['penalty']:
            # Initialize the model with the current parameters
            model = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=100)
            accuracies, confusion_matrices = n_fold_cross_validation(X, y, model, n=n)

            mean_accuracy = np.mean(accuracies)

            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_params = {'C': C, 'penalty': penalty}
                best_confusion_matrix = confusion_matrices
                best_accuracies = accuracies

    return best_params, best_score, best_confusion_matrix, best_accuracies

def main():
    # Load the data
    data = np.genfromtxt('mushrooms.csv', delimiter=',', dtype=str)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Perform 5-fold cross-validation with default model on training set
    default_model = LogisticRegression()
    accuracies, confusion_matrices = n_fold_cross_validation(X_train, y_train, default_model, n=5)

    # Print the results for the default model
    rounded_accuracies = [round(acc, 3) for acc in accuracies]  # Round accuracies to 3 decimals
    print("Accuracies for each fold (default model):", rounded_accuracies)
    print("Mean accuracy (default model):", round(np.mean(accuracies), 3))  # Round mean accuracy
    print("Confusion matrices for each fold (default model):")
    for cm in confusion_matrices:
        print(cm)

    # Standardize the features for training the default model on the training set
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    # Hyperparameters grid for custom grid search
    param_grid = {
        'C': [0.1, 0.5, 1, 5],
        'penalty': ['l1', 'l2']
    }

    # Perform custom grid search with 5-fold cross-validation on training set
    best_params, best_score, best_confusion_matrix, best_accuracies = custom_grid_search(X_train, y_train, param_grid, n=5)

    # Print the results from the grid search
    rounded_best_accuracies = [round(acc, 3) for acc in best_accuracies]  # Round best accuracies to 3 decimals
    print("Best parameters found:", best_params)
    print("Best mean accuracy from grid search:", round(best_score, 3))  # Round best mean accuracy
    print("Confusion matrices for each fold with the best parameters:")
    for cm in best_confusion_matrix:
        print(cm)

    # Train the best model on the entire training set
    best_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver='liblinear', max_iter=100)
    best_model.fit(X_train_std, y_train)

    # Evaluate overall training accuracy on the training set
    y_train_pred_best = best_model.predict(X_train_std)
    train_accuracy = accuracy_score(y_train, y_train_pred_best)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred_best)

    # Standardize the test set
    X_test_std = sc.transform(X_test)

    # Evaluate the best model on the test set
    y_test_pred_best = best_model.predict(X_test_std)
    best_test_accuracy = accuracy_score(y_test, y_test_pred_best)
    best_test_conf_matrix = confusion_matrix(y_test, y_test_pred_best)

    # Print overall training accuracy, confusion matrix, and test accuracy
    print("Best Model Overall Training Accuracy:", round(train_accuracy, 3))  # Round training accuracy to 3 decimals
    print("Best Model Training Confusion Matrix:\n", train_conf_matrix)
    print("Best Model Test Accuracy:", round(best_test_accuracy, 3))  # Round test accuracy to 3 decimals
    print("Best Model Test Confusion Matrix:\n", best_test_conf_matrix)

if __name__ == "__main__":
    main()
