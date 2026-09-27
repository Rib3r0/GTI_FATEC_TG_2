import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def treinar_modelo_restaurante():
    print("--- INICIANDO TREINAMENTO DO MODELO DE IA DE PREVISÃO ---")

    # 1. Carregar a tabela de movimentações de estoque
    df_mov = pd.read_csv('movimentacoes_estoque.csv')

    # 2. Filtrar apenas as movimentações de consumo (Saídas Reais)
    df_consumo = df_mov[df_mov['tipo_movimentacao'] == 'SAIDA_CONSUMO'].copy()

    # Extrair a data (YYYY-MM-DD) da coluna data_hora
    df_consumo['data'] = pd.to_datetime(df_consumo['data_hora']).dt.strftime('%Y-%m-%d')

    # 3. Agrupar o consumo total diário por produto e manter as variáveis do dia
    df_diario = df_consumo.groupby(
        ['data', 'produto_id', 'dia_semana', 'eh_feriado', 'clima', 'temp_media_c', 'evento_regiao']
    )['quantidade'].sum().reset_index()

    df_diario.rename(columns={'quantidade': 'qtd_consumida'}, inplace=True)

    print(f"Total de registros diários processados para treino: {len(df_diario)}")

    # 4. Transformar variáveis categóricas em numéricas (One-Hot Encoding)
    # Codifica 'dia_semana', 'clima' e o próprio 'produto_id'
    df_encoded = pd.get_dummies(df_diario, columns=['produto_id', 'dia_semana', 'clima'], drop_first=False)

    # 5. Separar Features (X) e Target (y)
    colunas_ignorar = ['data', 'qtd_consumida']
    X = df_encoded.drop(columns=colunas_ignorar)
    y = df_encoded['qtd_consumida']

    # 6. Dividir em Treino (80%) e Teste (20%) mantendo a ordem temporal
    limite_treino = int(len(df_encoded) * 0.8)
    X_train, X_test = X.iloc[:limite_treino], X.iloc[limite_treino:]
    y_train, y_test = y.iloc[:limite_treino], y.iloc[limite_treino:]

    # 7. Treinar o Modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # 8. Avaliar o Desempenho
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n✔ Treinamento concluído com sucesso!")
    print(f"-> Média do Erro Absoluto (MAE): {mae:.2f} kg/unid")
    print(f"-> Raiz do Erro Quadrático (RMSE): {rmse:.2f} kg/unid")

    # 9. Salvar o modelo treinado e a lista de colunas para uso na previsão
    pacote_modelo = {
        'modelo': modelo,
        'colunas_X': list(X.columns)
    }
    joblib.dump(pacote_modelo, 'modelo_demanda_restaurante.joblib')
    print("✔ Modelo salvo em 'modelo_demanda_restaurante.joblib'")

if __name__ == "__main__":
    treinar_modelo_restaurante()