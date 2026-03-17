import numpy as np
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

def model_choice(model_type):
    """Función para seleccionar el modelo a utilizar."""
    if model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=3)
    elif model_type == 'gaussianNB':
        return GaussianNB()
    else:
        raise ValueError("Ponlo bien")

def knn_or_nb(file, model_type='knn'):
    df = pd.read_csv(file, comment='@', header=None, delimiter=',')
    df_class = df.iloc[:, 0].values
    df_features = df.iloc[:, 1:].values
    np.set_printoptions(suppress=True)  # para no usar notación científica

    loo = LeaveOneOut()
    unique_classes = np.unique(df_class)
    y_true = []  
    y_classified = []

    model = model_choice(model_type)

    for train_index, test_index in loo.split(df_features):
        X_train, X_test = df_features[train_index], df_features[test_index]
        Y_train, Y_test = df_class[train_index], df_class[test_index]

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        y_classified.append(int(y_pred))
        y_true.append(int(Y_test))

    conf_matrix = confusion_matrix(y_true, y_classified, labels=unique_classes)
    return conf_matrix, unique_classes

def plot_confusion_matrix(matrix, labels, title="Matriz de confusión"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Clasificación')
    plt.ylabel('Realidad')
    plt.title(title)
    plt.show()

def matrixxx(model_type='knn'):
    matrix, unique_classes = knn_or_nb('wine.data', model_type)
    print(f"Matriz de Confusión Completa ({model_type}):")
    print(matrix)

    # Plotting the 3x3 confusion matrix
    plot_confusion_matrix(matrix, unique_classes, title=f"Matriz de confusión completa ({model_type})")

    confusion_matrices = np.zeros((len(matrix), 2, 2), dtype=int) 
    metrics = []
    for i, cls in enumerate(unique_classes):
        TP = matrix[i, i]
        FN = np.sum(matrix[i, :]) - TP
        FP = np.sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - (TP + FN + FP)
        confusion_matrices[i] = np.array([
            [TP, FN],
            [FP, TN]
        ])
        recall = TP / (TP + FN)
        spec = TN / (TN + FP)
        balanced_acc = (recall + spec) / 2
        precision = TP / (TP + FP)
        f1_score = 2 / ((1 / recall) + (1 / precision))
        metrics.append([recall, spec, balanced_acc, precision, f1_score])

    #métricas en tabulate
    print("\nMétricas por clase:")
    int_data = [[float(round(x, 5)) for x in row] for row in metrics]
    count = 0
    for row in int_data:
        row.insert(0, f"Clase{count+1}") 
        count += 1

    headers = ["Clases", "Sensibilidad", "Especificidad", "Balanced Acc", "Precisión", "F1 score"]
    print(tabulate(int_data, headers=headers, tablefmt="grid"))

    average_data = [['Macro'], ['Weighted'], ['Micro']]
    macro_avg = np.mean(metrics, axis=0)
    for element in macro_avg:
        average_data[0].append(element)

    class_counts = np.sum(matrix, axis=1)  # Cantidad de valoress de cada clase
    weighted_avg = np.average(metrics, axis=0, weights=class_counts)
    for element in weighted_avg:
        average_data[1].append(element)

    TP_total = np.sum(np.diagonal(matrix))  # Sumar todas las verdaderas positivas
    FN_total = np.sum(np.sum(matrix, axis=1)) - TP_total 
    FP_total = np.sum(np.sum(matrix, axis=0)) - TP_total
    TN_total = np.sum(matrix) - (TP_total + FN_total + FP_total)

    # micro 2x2
    recall_micro = TP_total / (TP_total + FN_total)
    precision_micro = TP_total / (TP_total + FP_total)
    f1_micro = 2 / ((1 / recall_micro) + (1 / precision_micro))
    micro_avg = [recall_micro, precision_micro, (recall_micro + precision_micro) / 2, precision_micro, f1_micro]
    average_data[2].extend(micro_avg)

    print("\nPromedios de las métricas (Macro, Weighted, Micro):")
    print(tabulate(average_data, tablefmt="grid"))

    # Plotting class confusion matrices
    for i, cls in enumerate(unique_classes):
        class_matrix = confusion_matrices[i]
        print(f"Matriz de confusión de clase {cls}:")
        print(class_matrix)
        plot_confusion_matrix(class_matrix, ["Positive", "Negative"], title=f"Matriz de confusión para Clase {cls}")

model_type = 'gaussianNB'  # Cambiar entre 'knn' y 'gaussianNB'
matrixxx(model_type)