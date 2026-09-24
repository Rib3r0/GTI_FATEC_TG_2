import pandas as pd
import joblib

def calcular_sugestao_compra():
    print("--- GERADOR DE PREVISÃO E SUGESTÃO DE COMPRA ---")

    # 1. Carregar o modelo salvo no Passo 2
    pacote = joblib.load('modelo_demanda_restaurante.joblib')
    modelo = pacote['modelo']
    colunas_X = pacote['colunas_X']

    # 2. Definir o cenário planejado para AMANHÃ (Exemplo: Sexta-feira ensolarada sem feriado)
    cenario_amanha = pd.DataFrame([{
        'eh_feriado': 0,
        'temp_media_c': 26.5,
        'evento_regiao': 1,
        'dia_semana_Sexta': 1,
        'clima_Ensolarado': 1
    }]).reindex(columns=colunas_X, fill_value=0)

    # 3. Realizar a previsão de demanda com a IA
    demanda_prevista_kg = modelo.predict(cenario_amanha)[0]

    # 4. Consultar o estoque atual em lotes_estoque.csv (Produto 101: Contra-Filé)
    df_lotes = pd.read_csv(r'.\arquivos_teste\lotes_estoque.csv')
    estoque_atual_kg = df_lotes[df_lotes['produto_id'] == 101]['quantidade_atual'].sum()

    # 5. Consultar a margem de segurança (Estoque Mínimo) em produtos.csv
    df_produtos = pd.read_csv(r'.\arquivos_teste\produtos.csv')
    estoque_minimo_kg = df_produtos[df_produtos['produto_id'] == 101]['estoque_minimo'].values[0]

    # 6. Calcular a Sugestão de Compras: (Demanda Prevista + Estoque Mínimo) - Estoque Atual
    sugestao_compra_kg = (demanda_prevista_kg + estoque_minimo_kg) - estoque_atual_kg
    sugestao_compra_final = max(0.0, sugestao_compra_kg) # Não permite valores negativos

    # 7. Exibir o Relatório do Sistema
    print("\n--- RESULTADO DA ANÁLISE DE ESTOQUE ---")
    print(f"Produto: Contra-Filé (ID 101)")
    print(f"Demanda Prevista (IA): {demanda_prevista_kg:.2f} kg")
    print(f"Estoque Disponível Hoje: {estoque_atual_kg:.2f} kg")
    print(f"Estoque Mínimo de Segurança: {estoque_minimo_kg:.2f} kg")
    print("---------------------------------------")
    print(f"👉 SUGESTÃO FINAL DE COMPRA: {sugestao_compra_final:.2f} kg")

if __name__ == "__main__":
    calcular_sugestao_compra()