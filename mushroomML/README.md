# 🍄 Mushroom Classification
## Overview
This project analyzes the Mushroom Classification Dataset from Kaggle to determine whether a mushroom is edible or poisonous based on various features. I implemented Logistic Regression and K-Nearest Neighbors (KNN) and compared their performance against baseline dummy classifiers.

## Dataset: [Mushroom Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification?resource=download)

I used the Mushroom Dataset from Kaggle, which contains **23 features** describing various characteristics such as cap, gill, stalk, veil, ring, bruises, odor, stalk shape, population, and habitat. The goal is to classify mushrooms as either **poisonous** or **edible**.

---

## Model

I applied **Logistic Regression** to the dataset using the default hyperparameters from Scikit-learn. The initial model achieved an accuracy score of **95.2%**.

### Accuracy

**0.952**

### Confusion Matrix

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 1206                 | 57                      |
| **Actual Poisonous (1)** | 60                   | 1115                    |

---

## Optimization

I fine-tuned the model using **Grid Search**, testing the following hyperparameters:
- **C values**: (0.1, 1, 10, 100)
- **Class weight**: None or balanced
- **Max iterations**: 100 or 200
- **Regularization (Penalty)**: L1 or L2
- **Solvers**: `liblinear` (useful for smaller datasets) and `saga` (better for larger datasets)

### Best Parameters Found

```json
{
  "C": 100,
  "class_weight": "balanced",
  "max_iter": 100,
  "penalty": "l2",
  "solver": "liblinear"
}
```

### Accuracy

**0.969**

### Confusion Matrix

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 1219                 | 44                      |
| **Actual Poisonous (1)** | 32                   | 1143                    |

---

## Comparison with K-Nearest Neighbors (KNN)

I implemented **KNN from scratch**, testing values of **K = 1 to 160**. The highest accuracy occurred for **K = 100-110**, all achieving **95% accuracy**. Accuracy declined beyond **K = 110**.

### Accuracy for K Values

| K-Value | Accuracy |
|---------|----------|
| 100-110 | 0.950    |
| 111     | 0.949    |
| 112     | 0.949    |
| 113     | 0.947    |
| 114-119 | 0.948    |

---

## Performance of Baseline Models

I evaluated two **Dummy Classifiers** (Stratified and Most Frequent) to compare model performance.

### Dummy Classifier (Stratified)

**Accuracy:** 0.494

### Dummy Classifier (Most Frequent)

**Accuracy:** 0.518

### Model Comparison Table

| Model                | Accuracy |
|----------------------|----------|
| **Logistic Regression** | 0.952    |
| **KNN**                  | 0.969    |
| **Dummy (Stratified)**   | 0.494    |
| **Dummy (Most Frequent)**| 0.518    |

---

## Confusion Matrices for All Models

### Dummy Classifier (Stratified)

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 632                  | 631                     |
| **Actual Poisonous (1)** | 602                  | 573                     |

### Dummy Classifier (Most Frequent)

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 1263                 | 0                       |
| **Actual Poisonous (1)** | 1175                 | 0                       |

### Logistic Regression Performance

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 1219                 | 44                      |
| **Actual Poisonous (1)** | 32                   | 1143                    |

### KNN Performance

|                      | Predicted Edible (0) | Predicted Poisonous (1) |
|----------------------|----------------------|-------------------------|
| **Actual Edible (0)**    | 588                  | 22                      |
| **Actual Poisonous (1)** | 100                  | 1728                    |

---

## Conclusion

Both **Logistic Regression** and **KNN** outperformed the dummy classifiers. While KNN achieved slightly higher accuracy (**96.9%**), Logistic Regression provided more **balanced classification**. Specifically, KNN **misclassified 100 poisonous mushrooms as edible**, whereas Logistic Regression only misclassified **32 poisonous mushrooms as edible**.

For real-world applications, **Logistic Regression** may be preferable since **fewer poisonous mushrooms are incorrectly classified as edible**, making it a **safer** choice despite KNN’s slightly higher accuracy.

