import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(train_data, train_labels, test_point, k):
    distances = []
    
    for i, row in train_data.iterrows():
        dist = euclidean_distance(row.values, test_point)
        distances.append((dist, train_labels[i]))

    #tuple (distance, class)
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for _, label in distances[:k]]
    
    # votes
    most_common = max(set(k_neighbors), key=k_neighbors.count)
    return most_common

def k_fold_cross_validation(df, k_folds, k_neighbors):
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]

    kf = LeaveOneOut() if k_folds == df.shape[0] else KFold(n_splits=k_folds, shuffle=True, random_state=4)
    
    y_true_all, y_pred_all = [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        for i, test_point in X_test.iterrows():
            predicted_class = knn(X_train, y_train, test_point.values, k=k_neighbors)
            y_pred_all.append(predicted_class)
            y_true_all.append(y_test[i])

    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, output_dict=False)
    print(report)
    #recall = report["macro avg"]["recall"]
    #specificity = report["macro avg"]["precision"]
    #balanced_acc = report["macro avg"]["f1-score"]
    recall=specificity=balanced_acc=0
    return recall, specificity, balanced_acc, cm

data_names = ['iris.data', 'led7digit.dat', 'heart.dat']

for data_name in data_names:
    df = pd.read_csv(data_name, comment='@', header=None, delimiter=None, engine='python')

    for k_folds in [5, len(df)]:
        k_neighbors = 5
        recall, specificity, balanced_acc, cm = k_fold_cross_validation(df, k_folds, k_neighbors)
        
        print(f"Archivo: {data_name}, k-folds: {k_folds}, k: {k_neighbors}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print('-' * 45)
