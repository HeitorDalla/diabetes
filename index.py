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
model = RandomForestClassifier(n_jobs=1,
                               random_state=42)

# Treinando o modelo com os dados
training = model.fit(X_train,
                     y_train)

# Previsão do modelo
X_test_preds = model.predict(X_test)

# Avaliação do modelo com parâmetros padrões
test_metrics = evaluate_metrics(y_test, X_test_preds)

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

# Fazendo previsão com o novo trainamento
rs_preds = rs_model.predict(X_test)

# Avaliando o novo modelo
rs_metrics = evaluate_metrics(y_test, rs_preds)

print("Os melhores parâmetros do modelo é: {}" .format(rs_model.best_params_))

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

# Treinar o modelo parametrizado
gs_training = gs_model.fit(X_train, y_train)

# Fazendo previsões com o modelo treinado
gs_preds = gs_model.predict(X_test)

# Avaliando o modelo treinado
gs_metrics = evaluate_metrics(y_test, gs_preds)

print("Os melhores parâmetros do modelo é: {}" .format(gs_model.best_params_))

# Salvar o modelo
pickle.dump(rs_model, open('final_model.pkl', 'wb'))