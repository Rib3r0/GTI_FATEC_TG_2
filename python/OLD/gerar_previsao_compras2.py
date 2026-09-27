import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def prever_demanda_periodo(dias_a_frente=7):
    """
    Realiza a previsão de demanda acumulada para os próximos N dias (7 para Semana, 30 para Mês)
    e calcula a sugestão de compra necessária para cobrir todo o período.
    """
    print(f"--- PREVISÃO DE DEMANDA PARA OS PRÓXIMOS {dias_a_frente} DIAS ---")

    # 1. Carregar o modelo treinado
    pacote = joblib.load('modelo_demanda_restaurante.joblib')
    modelo = pacote['modelo']
    colunas_X = pacote['colunas_X']

    # 2. Gerar o calendário do período futuro (Datas a partir de amanhã)
    data_hoje = datetime.now()
    dias_semana_nome = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    climas = ['Ensolarado', 'Chuvoso', 'Nublado', 'Frio']

    registros_futuros = []
    np.random.seed(123) # Para simular a previsão do tempo do período

    for i in range(1, dias_a_frente + 1):
        data_futura = data_hoje + timedelta(days=i)
        dia_num = data_futura.weekday()
        nome_dia = dias_semana_nome[dia_num]
        
        # Simulação de dados do clima/calendário previstos para cada dia
        clima_previsto = np.random.choice(climas, p=[0.5, 0.2, 0.2, 0.1])
        temp_prevista = float(np.random.uniform(18.0, 30.0))
        eh_feriado = 1 if (data_futura.day % 15 == 0) else 0
        evento_regiao = 1 if (dia_num in [4, 5] and np.random.rand() > 0.4) else 0

        # Monta a estrutura esperada pelo One-Hot Encoding
        linha_cenario = {
            'eh_feriado': eh_feriado,
            'temp_media_c': round(temp_prevista, 1),
            'evento_regiao': evento_regiao,
            f'dia_semana_{nome_dia}': 1,
            f'clima_{clima_previsto}': 1
        }
        registros_futuros.append((data_futura.strftime('%Y-%m-%d'), nome_dia, linha_cenario))

    # Convertendo para DataFrame estruturado com exatamente as colunas do treino
    df_cenario_futuro = pd.DataFrame([r[2] for r in registros_futuros]).reindex(columns=colunas_X, fill_value=0)

    # 3. Executar a IA para prever a demanda de cada dia do período
    previsoes_diarias = modelo.predict(df_cenario_futuro)

    # Criar um relatório dia a dia
    df_relatorio_diario = pd.DataFrame({
        'Data': [r[0] for r in registros_futuros],
        'Dia': [r[1] for r in registros_futuros],
        'Demanda_Prevista_kg': np.round(previsoes_diarias, 2)
    })

    # 4. Calcular a Demanda Acumulada do Período
    demanda_total_periodo_kg = df_relatorio_diario['Demanda_Prevista_kg'].sum()

    # 5. Consultar Estoque Atual e Estoque Mínimo (Exemplo com Contra-Filé / ID 101)
    df_lotes = pd.read_csv(r'arquivos_teste\lotes_estoque.csv')
    df_produtos = pd.read_csv(r'arquivos_teste\produtos.csv')

    estoque_atual_kg = df_lotes[df_lotes['produto_id'] == 101]['quantidade_atual'].sum()
    estoque_minimo_kg = df_produtos[df_produtos['produto_id'] == 101]['estoque_minimo'].values[0]

    # 6. Calcular a Sugestão Final de Compra para o Período
    sugestao_compra_kg = max(0.0, (demanda_total_periodo_kg + estoque_minimo_kg) - estoque_atual_kg)

    # Exibição dos resultados
    print("\n--- PREVISÃO DIA A DIA DO PERÍODO ---")
    print(df_relatorio_diario.to_string(index=False))

    print("\n---------------------------------------")
    print(f"Demanda Total Prevista ({dias_a_frente} dias): {demanda_total_periodo_kg:.2f} kg")
    print(f"Estoque Disponível Hoje: {estoque_atual_kg:.2f} kg")
    print(f"Estoque Mínimo de Segurança: {estoque_minimo_kg:.2f} kg")
    print("---------------------------------------")
    print(f"👉 SUGESTÃO DE COMPRA PARA O PERÍODO: {sugestao_compra_kg:.2f} kg")

if __name__ == "__main__":
    # Para prever a semana, use 7. Para o mês, basta alterar para 30
    prever_demanda_periodo(dias_a_frente=7)