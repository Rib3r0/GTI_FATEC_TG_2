# Projeto feito apenas para entender conceitos de machine learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prever se um cliente vai cancelar o serviço ou não
# dados = {
#     'meses_como_cliente': [2, 35, 1, 48, 5, 60, 3, 24],
#     'gasto_mensal': [70.5, 110.0, 50.0, 95.0, 80.0, 105.0, 65.0, 85.0],
#     'suporte_chamados': [5, 1, 6, 0, 4, 1, 5, 2],
#     'cancelou': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Cancelou, 0 = Permanece
# }

dados = {
    'meses_como_cliente': [
        2, 35, 1, 48, 5, 60, 3, 24, 12, 54, 
        4, 18, 2, 36, 8, 72, 6, 15, 1, 42, 
        10, 28, 3, 50, 7, 14, 9, 30, 2, 66, 70
    ],
    'gasto_mensal': [
        70.5, 110.0, 50.0, 95.0, 80.0, 105.0, 65.0, 85.0, 75.0, 115.0,
        90.0, 68.0, 55.0, 100.0, 82.0, 120.0, 78.0, 62.0, 48.0, 98.0,
        88.0, 72.0, 60.0, 108.0, 79.0, 66.0, 84.0, 92.0, 52.0, 112.0, 85.0
    ],
    'suporte_chamados': [
        5, 1, 6, 0, 4, 1, 5, 2, 3, 0,
        6, 2, 4, 1, 3, 0, 5, 2, 7, 1,
        4, 2, 5, 0, 3, 3, 2, 1, 6, 0, 0
    ],
    'cancelou': [
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1
    ]  # 1 = Cancelou, 0 = Permanece
}


df = pd.DataFrame(dados)
print(f"Total de registros: {len(df)}")
print(df.head(31))

X = df[['meses_como_cliente', 'gasto_mensal', 'suporte_chamados']]

y = df['cancelou']

# 80% treino , 20% teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42
)

modelo = DecisionTreeClassifier(random_state=42)

modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia do Modelo: {acuracia * 100:.2f}%\n")

importancia = pd.DataFrame({
    'Variavel': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("Importância das Variáveis para a Análise:")
print(importancia)