import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prever se um cliente vai cancelar o serviço ou não
dados = {
    'meses_como_cliente': [2, 35, 1, 48, 5, 60, 3, 24],
    'gasto_mensal': [70.5, 110.0, 50.0, 95.0, 80.0, 105.0, 65.0, 85.0],
    'suporte_chamados': [5, 1, 6, 0, 4, 1, 5, 2],
    'cancelou': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Cancelou (Churn), 0 = Permanece
}

df = pd.DataFrame(dados)

print(df.head())

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