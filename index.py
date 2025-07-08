import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle

# Função para avaliar o modelo
def evaluate_metrics (y_true, y_prev):
    accuracy = accuracy_score(y_true, y_prev)
    precision = precision_score(y_true, y_prev)
    recall = recall_score(y_true, y_prev)

    metrics = {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2)
    }

    print("A acurácia é: {:.2f}" .format(accuracy))
    print("A precision é: {:.2f}" .format(precision))
    print("O recall é: {:.2f}" .format(recall))

    return metrics

# Estruturação dos Dados
df = pd.read_csv('data/diabetes.csv', sep=',', encoding='utf-8')

print(df.dtypes)
print(df.describe())
print(df.isna().sum())

print(df['Outcome'].value_counts()) # ver contagem total de cada opção da coluna (importante para maior coleta de dados)

# Dividir os dados em colunas dependentes e independentes
X = df.drop(columns=['Outcome'], axis=1)
y = df['Outcome']

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Escolher modelo para treinamento dos dados
model = RandomForestClassifier(max_depth=10,
                               min_samples_leaf=5,
                               n_estimators=100,
                               n_jobs=-1,
                               random_state=42)

# Treinando o modelo com os dados
training = model.fit(X_train,
                     y_train)

# Previsão e avaliação do modelo de treinamento
print("As métricas do modelo de treinamento é: ")
y_train_preds = model.predict(X_train)
train_metrics = evaluate_metrics(y_train, y_train_preds)

print("------------------")

# Previsão e avaliação do modelo de teste
print("As métricas do modelo de teste é: ")
y_test_preds = model.predict(X_test)
test_metrics = evaluate_metrics(y_test, y_test_preds)

# Melhorando os parâmetros
grid = {
    "max_depth": [None, 5, 10, 20, 30],
    "max_features": ['sqrt', 'log2'],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 4, 6],
    "n_estimators": [10, 100, 200, 500]
}

# Definindo o novo modelo com os parâmetros
rs_model = RandomizedSearchCV(model,
                              param_distributions=grid,
                              n_iter=10,
                              cv=5,
                              verbose=2)

# Treinar o novo modelo
rs_training = rs_model.fit(X_train, y_train)

# Melhor modelo do RandomizedSearch
best_rs_model = rs_model.best_estimator_

# Previsão e avaliação do modelo de treinamento do RandomizedSearch
print("Avaliação do melhor modelo de treinamento do RandomizedSearch é: ")
y_train_preds_rs = best_rs_model.predict(X_train)
train_metrics_rs = evaluate_metrics(y_train, y_train_preds_rs)

print("------------------")

# Previsão e avaliação do modelo de teste do RandomizedSearch
print("Avaliação do melhor modelo de teste do RandomizedSearch é: ")
y_test_preds_rs = best_rs_model.predict(X_test)
test_metrics_rs = evaluate_metrics(y_test, y_test_preds_rs)

# Definindo um novo modelo a partir dos melhores hiperparâmetros
grid2 = {
    'n_estimators': [100, 200],
    'min_samples_split': [6],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'max_depth': [5, 10, 20]
}

gs_model = GridSearchCV(model,
                        param_grid=grid2,
                        cv=5,
                        verbose=2)

# Treinar o novo modelo
gs_training = gs_model.fit(X_train, y_train)

# Melhor modelo do GridSearch
best_gs_model = gs_model.best_estimator_

# Previsão e avaliação do modelo de treinamento do GridSearch
print("Avaliação do melhor modelo de treinamento do GridSearch é: ")
y_train_preds_gs = best_gs_model.predict(X_train)
train_metrics_gs = evaluate_metrics(y_train, y_train_preds_gs)

print("------------------")

# Previsão e avaliação do modelo de teste do GridSearch
print("Avaliação do melhor modelo de teste do GridSearch é: ")
y_test_preds_gs = best_gs_model.predict(X_test)
test_metrics_gs = evaluate_metrics(y_test, y_test_preds_gs)

# Salvar o modelo
pickle.dump(best_gs_model, open('final_model.pkl', 'wb'))