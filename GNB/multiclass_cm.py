import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(archivo):
    datos = pd.read_csv(archivo, comment='@', header=None)
    return datos.iloc[:, :-1].values, datos.iloc[:, -1].values

def entrenar_modelo(X_entrenamiento, y_entrenamiento):
    clases = np.unique(y_entrenamiento)
    parametros = {}
    
    for clase in clases:
        X_clase = X_entrenamiento[y_entrenamiento == clase]
        media = np.mean(X_clase, axis=0)
        desviacion = np.std(X_clase, axis=0)
        prior = len(X_clase) / len(X_entrenamiento)
        parametros[clase] = {'media': media, 'std': desviacion, 'log_prior': np.log(prior)}
    
    return parametros

def calcular_log_probabilidad(x, media, std):
    return norm.logpdf(x, loc=media, scale=std)

def predecir(X_prueba, parametros):
    clases = list(parametros.keys())
    predicciones = []
    
    for x in X_prueba:
        log_probabilidades = []
        
        for clase in clases:
            log_verosimilitud = np.sum(calcular_log_probabilidad(x, parametros[clase]['media'], parametros[clase]['std']))
            log_posterior = log_verosimilitud + parametros[clase]['log_prior']
            log_probabilidades.append(log_posterior)
        
        clase_predicha = clases[np.argmax(log_probabilidades)]
        predicciones.append(clase_predicha)
    
    return np.array(predicciones)

def evaluar_modelo(X, y, validador):
    y_real, y_predicho = [], []
    
    for indice_entrenamiento, indice_prueba in validador.split(X):
        X_entrenamiento, X_prueba = X[indice_entrenamiento], X[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]
        
        parametros = entrenar_modelo(X_entrenamiento, y_entrenamiento)
        predicciones = predecir(X_prueba, parametros)
        
        y_real.extend(y_prueba)
        y_predicho.extend(predicciones)
    
    return confusion_matrix(y_real, y_predicho, labels=np.unique(y))

def calcular_metricas(matriz_confusion):
    n_clases = matriz_confusion.shape[0]
    metricas = []
    
    for i in range(n_clases):
        TP = matriz_confusion[i,i]
        FN = np.sum(matriz_confusion[i,:]) - TP
        FP = np.sum(matriz_confusion[:,i]) - TP
        TN = np.sum(matriz_confusion) - (TP + FN + FP)
        
        sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0
        especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0
        exactitud_balanceada = (sensibilidad + especificidad) / 2
        
        metricas.append({
            'Clase': i,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Balanced Accuracy': exactitud_balanceada,
            'Sensibilidad': sensibilidad,
            'Especificidad': especificidad
        })
    
    return metricas

def mostrar_resultados(matriz_confusion, nombre_metodo):
    metricas = calcular_metricas(matriz_confusion)
    
    print(f"\nResultados con {nombre_metodo}")
    print(f"\nMatriz de Confusión Global:")
    print(matriz_confusion)
    
    # Mostrar métricas por clase con valores de confusión
    print("\nMétricas por Clase:")
    tabla = []
    for m in metricas:
        tabla.append([
            f"Clase {m['Clase']}",
            m['TP'],
            m['TN'],
            m['FP'],
            m['FN'],
            f"{m['Balanced Accuracy']:.4f}",
            f"{m['Sensibilidad']:.4f}",
            f"{m['Especificidad']:.4f}"
        ])
    
    print(tabulate(tabla, 
                 headers=["Clase", "TP", "TN", "FP", "FN", "Balanced Acc", "Sensibilidad", "Especificidad"],
                 tablefmt="grid"))
    
    # promedios
    balanced_acc_avg = np.mean([m['Balanced Accuracy'] for m in metricas])
    sensibilidad_avg = np.mean([m['Sensibilidad'] for m in metricas])
    especificidad_avg = np.mean([m['Especificidad'] for m in metricas])
    
    print(f"\nPromedios Macro:")
    print(f"Balanced Accuracy: {balanced_acc_avg:.4f}")
    print(f"Sensibilidad: {sensibilidad_avg:.4f}")
    print(f"Especificidad: {especificidad_avg:.4f}")
    
    # Gráfico de matriz de confusión
    plt.figure(figsize=(8,6))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Clase {i}" for i in range(len(metricas))],
                yticklabels=[f"Clase {i}" for i in range(len(metricas))])
    plt.title(f"Matriz de Confusión\n{nombre_metodo}")
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

archives = ['iris.data', 'heart.dat', 'led7digit.dat']
for archive in archives:
    X, y = cargar_datos(archive)

    print("="*50)
    print(f"NAIVE BAYES GAUSSIANO DE {archive}")
    print("="*50)

    #  Leave-One-Out
    matriz_loo = evaluar_modelo(X, y, LeaveOneOut())
    mostrar_resultados(matriz_loo, "Leave-One-Out")

    # 10-Fold
    matriz_kfold = evaluar_modelo(X, y, KFold(n_splits=10))
    mostrar_resultados(matriz_kfold, "10-Fold CV")