import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def treinar_modelo():
    print("Carregando histórico de vendas para treinar a IA...")

    df_vendas = pd.read_csv(r'.\arquivos_teste\historico_vendas_diario.csv')

    df_encoded = pd.get_dummies(df_vendas, columns=['dia_semana', 'clima'], drop_first=False)

    colunas_ignorar = ['vendas_id','data', 'produto_id', 'qtd_vendida']
    X = df_encoded.drop(columns=colunas_ignorar)
    y = df_encoded['qtd_vendida']

    tamanho_treino = int(len(df_vendas) * 0.8)
    X_treino, X_teste = X.iloc[:tamanho_treino], X.iloc[tamanho_treino:]
    y_treino, y_teste = y.iloc[:tamanho_treino], y.iloc[tamanho_treino:]

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    previsoes = modelo.predict(X_teste)
    mae = mean_absolute_error(y_teste, previsoes)
    rmse = np.sqrt(mean_squared_error(y_teste, previsoes))

    print("\n--- DESEMPENHO DA IA ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f} kg")
    print(f"Erro Quadrático Médio (RMSE): {rmse:.2f} kg")

    pacote_salvamento = {
        'modelo': modelo,
        'colunas_X': list(X.columns)
    }
    joblib.dump(pacote_salvamento, 'modelo_demanda_restaurante.joblib')
    print("\n✔ Modelo treinado e salvo com sucesso em 'modelo_demanda_restaurante.joblib'!")

if __name__ == "__main__":
    treinar_modelo()