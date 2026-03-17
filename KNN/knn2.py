import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(train_data, train_labels, test_point, k):
    distances = []
    
    # Calcular distancias desde el punto de prueba a todos los puntos de entrenamiento
    for i, row in train_data.iterrows():
        dist = euclidean_distance(row.values, test_point)
        distances.append((dist, train_labels[i]))
    
    #tuple with order of distance (distancia, etiqueta)
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for _, label in distances[:k]]
    
    class_counts = {}
    for label in k_neighbors:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    # best class
    most_common = max(class_counts, key=class_counts.get)
    return most_common, class_counts

def k_fold_cross_validation(df, k_folds, k_neighbors):
    X = df[df.columns[:-1]]  # features
    y = df.iloc[:, -1]  # class

    kf = LeaveOneOut() if k_folds == df.shape[0] else KFold(n_splits=k_folds, shuffle=True, random_state=4)
    all_TP = all_TN = all_FP = all_FN = 0

    # Para cada fold, definimos particiones de entrenamiento y prueba
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_pred = []
        for i, test_point in X_test.iterrows():
            predicted_class, class_counts = knn(X_train, y_train, test_point.values, k=k_neighbors)
            y_pred.append(predicted_class)

        cm = confusion_matrix(y_test, y_pred, labels=list(y.unique().tolist()))
        # Para cada clase, calculamos TP, TN, FP, FN
        for i in range(len(cm)):
            if len(cm) == 2:
                TP = cm[0, 0]
                FP = cm[1, 0]
                FN = cm[0, 1]
                TN = cm[1, 1]
                all_TP += TP
                all_TN += TN
                all_FP += FP
                all_FN += FN
                break
            else:
                TP = cm[i, i]
                FP = sum(cm[:, i]) - TP
                FN = sum(cm[i, :]) - TP
                TN = np.sum(cm) - (TP + FN + FP)

            all_TP += TP
            all_TN += TN
            all_FP += FP
            all_FN += FN

    #métricas de evaluación
    recall = all_TP / (all_TP + all_FN)
    specificity = all_TN / (all_TN + all_FP)
    balance_acc = (recall + specificity) / 2
    valid_metrics = [recall, specificity, balance_acc]
    
    return valid_metrics, all_TP, all_TN, all_FP, all_FN, k_folds, k_neighbors

data_names = ['iris.data', 'led7digit.dat', 'heart.dat']


for data_name in data_names:
    df = pd.read_csv(data_name, comment='@', header=None, delimiter=',')

    for i in range(2):
        if i == 0:
            k_folds = 5
        if i == 1:
            leng = len(df)
            k_folds = leng
        
        valid_metrics, all_TP, all_TN, all_FP, all_FN, k_folds, k_neighbors= k_fold_cross_validation(df, k_folds, k_neighbors=5)

        # Mostrar resultados
        print(f"Archivo: {data_name}, k-folds: {k_folds}, k: {k_neighbors}")
        print(f"TP: {all_TP}")
        print(f"TN: {all_TN}")
        print(f"FP: {all_FP}")
        print(f"FN: {all_FN}")
        print(f"Recall: {valid_metrics[0]}")
        print(f"Specificity: {valid_metrics[1]}")
        print(f"Balanced Accuracy: {valid_metrics[2]}")
        print('---' * 15)

