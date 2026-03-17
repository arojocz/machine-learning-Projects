
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ['iris.data', 'heart.dat', 'led7digit.dat']

def read_dataset(dataset,  delimiter=',', header=None):
        df = pd.read_csv(dataset, comment='@')

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # binary classes only
        if dataset == 'iris.data':
            mask = y != 'Iris-setosa'
            X = X[mask]
            y = y[mask]
            y = np.where(y == 'Iris-versicolor', 0, 1)
            #y -= 1  # 0 y 1
        else:
            mask = (y > 0) & (y < 3)
            X = X[mask]
            y = y[mask]
            y -= 1  # 0 y 1

        # Hold-out 70/30 stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3 , stratify=y, random_state=42
        )
        return X, y, X_train, X_test, y_train, y_test

def gini(X, y):
    #sort each column and average of every 2 consecutive values
    min_values = np.array([])
    best_thresholds = []
    for i in range(X.shape[1]):
        gini_list = []
        column = X[:,i]
        index_sort = np.argsort(column) #sorted indexes
        column_sorted = column[index_sort]
        y_sorted = y[index_sort]
        averages = (column_sorted[1:] + column_sorted[:-1]) / 2
        for average in averages:
            left_leaf = column_sorted < average
            right_leaf = column_sorted >= average
            
            left_leaf = y_sorted[left_leaf]
            right_leaf = y_sorted[right_leaf]
             
            #calculate gini
            mask1 = left_leaf == 0
            mask2 = right_leaf == 0

            if len(left_leaf) > 0:
                p1 = len(left_leaf[mask1]) / len(left_leaf)
                p2 = len(left_leaf[~mask1]) / len(left_leaf)
                gini_1 = 1 - p1**2 - p2**2
            else:
                gini_1 = 1

            if len(right_leaf) > 0:
                p1 = len(right_leaf[mask2]) / len(right_leaf)
                p2 = len(right_leaf[~mask2]) / len(right_leaf)
                gini_2 = 1 - p1**2 - p2**2
            else:
                gini_2 = 1

            gini = gini_1*(len(left_leaf)/len(column_sorted)) + gini_2*(len(right_leaf)/len(column_sorted))
            gini_list.append(gini)

        gini_array = np.array(gini_list)
        min_gini = np.argmin(gini_array)
        best_thresholds.append(averages[min_gini])
        min_values = np.append(min_values, gini_array[min_gini])

    best_column = np.argmin(min_values)
    best_threshold = best_thresholds[best_column]
    return best_column, best_threshold, min_values[best_column]

def tree(X, y, depth = 0):
    # base case
    if len(set(y)) == 1 or depth == 2:
        # leaf with majority class
        return {'type': 'leaf', 'class': y[0], 'gini': 0}  # gini = 0
    
    # best división with Gini
    best_column, best_threshold, min_gini = gini(X, y)

    # branches
    left_mask = X[:, best_column] < best_threshold
    right_mask = ~left_mask

    # base case 2
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return {'type': 'leaf', 'class': np.bincount(y).argmax(), 'gini': min_gini}
    
    # divide the dataset
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    
    # recursively create subtrees
    left_subtree = tree(X_left, y_left, depth = depth +1)
    right_subtree = tree(X_right, y_right, depth= depth + 1)

    # create the node with divisions, subtres and gini
    return {'type': 'internal',
            'column': best_column,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'gini': min_gini
    }

def print_tree(node, depth=0):
    indent = "  " * depth
    if node['type'] == 'leaf':
        print(f"{indent}Leaf: class = {node['class']} (gini = {node['gini']:.4f})")
    else:
        print(f"{indent}Node: X[{node['column']}] < {node['threshold']:.3f} (gini = {node['gini']:.4f})")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)


def decision_tree(tree_model, x):
    while tree_model['type'] != 'leaf':
        if x[tree_model['column']] < tree_model['threshold']:
            tree_model = tree_model['left']
        else:
            tree_model = tree_model['right']
    return tree_model['class']

for i in datasets:
    print(f'Clasificación para {i}')
    X, y, X_train, X_test, y_train, y_test = read_dataset(i)
    #gini(X_train, y_train)
    print('-'*100)
    tree_built = tree(X_train, y_train)
    #pred = decision_tree(tree_built, X_test[0])
    y_pred = [decision_tree(tree_built, x) for x in X_test]
    counter = 0
    for element in y_pred:
        print(f"Clasificador: {element}, Verdadero: {y_test[counter]}")
        counter += 1

    balanced = balanced_accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    print('')
    print(f"Balanced Accuracy: {balanced:.4f}")
    print(f"Sensibilidad (clase 1): {sensitivity:.4f}")
    print(f"Especificidad (clase 0): {specificity:.4f}")
    print('')
    print_tree(tree_built)
    print('='*90)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clase 0', 'Clase 1'], 
                yticklabels=['Clase 0', 'Clase 1'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor verdadero')
    plt.title(f'Matriz de Confusión de {i}')
    plt.show()