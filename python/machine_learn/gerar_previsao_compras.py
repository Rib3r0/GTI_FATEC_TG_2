import pandas as pd
import numpy as np
import holidays
import joblib
from datetime import datetime, timedelta

def executar_previsao_e_compras(dias_horizonte=30):
    """
    Parâmetro 'dias_horizonte':
    - 1   -> Previsão para o Próximo Dia
    - 7   -> Previsão para a Próxima Semana
    - 30  -> Previsão para o Próximo Mês
    - 365 -> Previsão para o Próximo Ano
    """
    print(f"--- MÓDULO DE PREVISÃO E COMPRAS (HORIZONTE: {dias_horizonte} DIAS) ---")

    feriados_br = holidays.BR(state='SP')

    # 1. Carregar o modelo treinado e suas colunas
    pacote = joblib.load('modelo_demanda_restaurante.joblib')
    modelo = pacote['modelo']
    colunas_X = pacote['colunas_X']

    # 2. Carregar produtos cadastrados
    df_produtos = pd.read_csv('produtos.csv')
    df_lotes = pd.read_csv('lotes_estoque.csv')

    # 3. Gerar o calendário futuro do período informado
    data_inicio = datetime.now()
    dias_semana_nome = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    climas_possiveis = ['Ensolarado', 'Chuvoso', 'Nublado', 'Frio']

    np.random.seed(100) # Semente para simulação do clima futuro
    registros_futuros = []

    for dia_idx in range(1, dias_horizonte + 1):
        data_futura = data_inicio + timedelta(days=dia_idx)
        nome_dia = dias_semana_nome[data_futura.weekday()]
        mes = data_futura.month
        dia_mes = data_futura.day

        eh_feriado = 1 if data_futura in feriados_br else 0
        clima = np.random.choice(climas_possiveis, p=[0.5, 0.2, 0.2, 0.1])
        temp = float(np.random.uniform(18.0, 32.0))
        evento_regiao = 1 if (data_futura.weekday() in [4, 5] and np.random.rand() > 0.5) else 0

        # Para cada produto, monta o cenário de entrada do modelo
        for p_id in df_produtos['produto_id']:
            cenario_dia = {
                'eh_feriado': eh_feriado,
                'temp_media_c': round(temp, 1),
                'evento_regiao': evento_regiao,
                f'produto_id_{p_id}': 1,
                f'dia_semana_{nome_dia}': 1,
                f'clima_{clima}': 1
            }
            registros_futuros.append({
                'produto_id': p_id,
                'data': data_futura.strftime('%Y-%m-%d'),
                'cenario': cenario_dia
            })

    # 4. Formatar a entrada do modelo com todas as colunas esperadas
    df_cenario_completo = pd.DataFrame([r['cenario'] for r in registros_futuros]).reindex(columns=colunas_X, fill_value=0)

    # 5. Fazer a predição da demanda diária
    previsoes_diarias = modelo.predict(df_cenario_completo)

    # Adicionar o resultado predito aos registros
    for idx, reg in enumerate(registros_futuros):
        reg['demanda_prevista'] = previsoes_diarias[idx]

    df_previsoes_detalhadas = pd.DataFrame(registros_futuros)

    # 6. Agrupar a demanda total do período por produto
    demanda_acumulada = df_previsoes_detalhadas.groupby('produto_id')['demanda_prevista'].sum().reset_index()

    # 7. Cruzar com Estoque Atual e Estoque Mínimo para gerar a Ordem de Compra
    estoque_atual = df_lotes.groupby('produto_id')['quantidade_atual'].sum().reset_index()

    relatorio_compras = pd.merge(df_produtos, demanda_acumulada, on='produto_id', how='left')
    relatorio_compras = pd.merge(relatorio_compras, estoque_atual, on='produto_id', how='left').fillna({'quantidade_atual': 0})

    # Regra de negócio: Sugestão = (Demanda Prevista no Período + Estoque Mínimo) - Estoque Atual
    relatorio_compras['sugestao_compra'] = (
        (relatorio_compras['demanda_prevista'] + relatorio_compras['estoque_minimo']) - relatorio_compras['quantidade_atual']
    )
    relatorio_compras['sugestao_compra'] = relatorio_compras['sugestao_compra'].apply(lambda x: max(0.0, round(x, 2)))
    relatorio_compras['demanda_prevista'] = relatorio_compras['demanda_prevista'].round(2)

    # 8. Exibir o Relatório Final de Compras
    print(f"\n================ RELATÓRIO DE COMPRAS ({dias_horizonte} DIAS) ================")
    cols_exibir = ['produto_id', 'nome', 'unidade_medida', 'demanda_prevista', 'quantidade_atual', 'estoque_minimo', 'sugestao_compra']
    
    df_exibicao = relatorio_compras[cols_exibir].rename(columns={
        'produto_id': 'ID',
        'nome': 'Produto',
        'unidade_medida': 'Un',
        'demanda_prevista': f'Demanda ({dias_horizonte}d)',
        'quantidade_atual': 'Estoque Atual',
        'estoque_minimo': 'Est. Mínimo',
        'sugestao_compra': '👉 Sugestão Compra'
    })

    print(df_exibicao.to_string(index=False))

    # 9. Gravar resultado final no CSV 'previsoes_demanda_compras.csv'
    relatorio_salvar = relatorio_compras[['produto_id', 'demanda_prevista', 'quantidade_atual', 'sugestao_compra']].copy()
    relatorio_salvar.rename(columns={
        'demanda_prevista': 'demanda_prevista_periodo',
        'quantidade_atual': 'estoque_atual_kg',
        'sugestao_compra': 'sugestao_compra_kg'
    }, inplace=True)
    relatorio_salvar['horizonte_dias'] = dias_horizonte
    relatorio_salvar['gerado_em'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    relatorio_salvar.to_csv('previsoes_demanda_compras.csv', index=False, encoding='utf-8-sig')
    print("\n✔ Relatório gravado com sucesso em 'previsoes_demanda_compras.csv'.")

if __name__ == "__main__":
    # Teste alterando o número de dias: 1 (Dia), 7 (Semana), 30 (Mês) ou 365 (Ano)
    executar_previsao_e_compras(dias_horizonte=7)