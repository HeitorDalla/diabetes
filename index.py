import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Função para avaliar o modelo
def evaluate_metrics (y_true, y_prev):
    accuracy = accuracy_score(y_true, y_prev)
    precision = precision_score(y_true, y_prev)
    recall = recall_score(y_true, y_prev)

    metrics = {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'r2': round(recall, 2)
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

