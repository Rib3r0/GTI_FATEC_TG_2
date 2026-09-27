import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def gerar_massa_testes_restaurante():
    print("Iniciando a geração da massa de testes com Tabela de Movimentações...")

    # ==========================================
    # 1. CADASTRO DE PRODUTOS (10 ITENS)
    # ==========================================
    produtos_data = [
        {'produto_id': 101, 'nome': 'Contra-Filé', 'categoria': 'Carnes', 'unidade_medida': 'kg', 'estoque_minimo': 20.0, 'base_util': 15.0, 'base_fds': 40.0},
        {'produto_id': 102, 'nome': 'Peito de Frango', 'categoria': 'Carnes', 'unidade_medida': 'kg', 'estoque_minimo': 15.0, 'base_util': 18.0, 'base_fds': 25.0},
        {'produto_id': 103, 'nome': 'Queijo Mozarela', 'categoria': 'Laticínios', 'unidade_medida': 'kg', 'estoque_minimo': 8.0, 'base_util': 6.0, 'base_fds': 18.0},
        {'produto_id': 104, 'nome': 'Creme de Leite', 'categoria': 'Laticínios', 'unidade_medida': 'L', 'estoque_minimo': 5.0, 'base_util': 4.0, 'base_fds': 10.0},
        {'produto_id': 105, 'nome': 'Alface Crespa', 'categoria': 'Hortifrúti', 'unidade_medida': 'Un', 'estoque_minimo': 15.0, 'base_util': 12.0, 'base_fds': 15.0},
        {'produto_id': 106, 'nome': 'Tomate Italiano', 'categoria': 'Hortifrúti', 'unidade_medida': 'kg', 'estoque_minimo': 10.0, 'base_util': 8.0, 'base_fds': 14.0},
        {'produto_id': 107, 'nome': 'Arroz Arbóreo', 'categoria': 'Grãos', 'unidade_medida': 'kg', 'estoque_minimo': 10.0, 'base_util': 10.0, 'base_fds': 12.0},
        {'produto_id': 108, 'nome': 'Feijão Preto', 'categoria': 'Grãos', 'unidade_medida': 'kg', 'estoque_minimo': 12.0, 'base_util': 14.0, 'base_fds': 8.0},
        {'produto_id': 109, 'nome': 'Cerveja Artesanal', 'categoria': 'Bebidas', 'unidade_medida': 'L', 'estoque_minimo': 30.0, 'base_util': 12.0, 'base_fds': 65.0},
        {'produto_id': 110, 'nome': 'Azeite Extra Virgem', 'categoria': 'Temperos', 'unidade_medida': 'L', 'estoque_minimo': 4.0, 'base_util': 1.5, 'base_fds': 3.0}
    ]

    df_produtos_config = pd.DataFrame(produtos_data)
    cols_produtos = ['produto_id', 'nome', 'categoria', 'unidade_medida', 'estoque_minimo']
    df_produtos_config[cols_produtos].to_csv('produtos.csv', index=False, encoding='utf-8-sig')
    print("✔ [1/3] 'produtos.csv' gerado.")

    # ==========================================
    # 2. CONTROLE DE LOTES DE ESTOQUE
    # ==========================================
    lotes_data = []
    lote_counter = 1

    for _, prod in df_produtos_config.iterrows():
        p_id = prod['produto_id']
        
        # Lote 1 para todos
        lotes_data.append({
            'lote_id': f'L{lote_counter:03d}',
            'produto_id': p_id,
            'codigo_lote': f'LOT-202609-{p_id}-A',
            'quantidade_inicial': round(prod['estoque_minimo'] * 3, 2),
            'quantidade_atual': round(prod['estoque_minimo'] * 1.5, 2),
            'data_validade': '2026-10-20',
            'data_entrada': '2026-09-10'
        })
        lote_counter += 1

        # Lote 2 para produtos ímpares
        if p_id % 2 != 0:
            lotes_data.append({
                'lote_id': f'L{lote_counter:03d}',
                'produto_id': p_id,
                'codigo_lote': f'LOT-202609-{p_id}-B',
                'quantidade_inicial': round(prod['estoque_minimo'] * 2, 2),
                'quantidade_atual': round(prod['estoque_minimo'] * 2, 2),
                'data_validade': '2026-11-05',
                'data_entrada': '2026-09-18'
            })
            lote_counter += 1

    df_lotes = pd.DataFrame(lotes_data)
    df_lotes.to_csv('lotes_estoque.csv', index=False, encoding='utf-8-sig')
    print(f"✔ [2/3] 'lotes_estoque.csv' gerado com {len(df_lotes)} lotes.")

    # ==========================================
    # 3. MOVIMENTAÇÕES DE ESTOQUE (Entradas, Saídas e Desperdício)
    # ==========================================
    np.random.seed(42)
    data_inicio = datetime(2025, 10, 1)
    dias_semana_nome = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    climas_possiveis = ['Ensolarado', 'Chuvoso', 'Nublado', 'Frio']
    motivos_desperdicio = ['Validade / Avaria', 'Erro de Preparo', 'Quebra de Embalagem', 'Sobras de Prato']

    movimentacoes_data = []
    mov_counter = 1000

    # Registrar entradas iniciais dos lotes na tabela de movimentações
    for _, lote in df_lotes.iterrows():
        movimentacoes_data.append({
            'movimentacao_id': f'M{mov_counter}',
            'lote_id': lote['lote_id'],
            'produto_id': lote['produto_id'],
            'tipo_movimentacao': 'ENTRADA',
            'quantidade': lote['quantidade_inicial'],
            'motivo_desperdicio': None,
            'usuario_id': 1, # ID do Usuário Adm/Compras
            'data_hora': f"{lote['data_entrada']} 08:00:00",
            'dia_semana': 'Quinta',
            'eh_feriado': 0,
            'clima': 'Ensolarado',
            'temp_media_c': 24.0,
            'evento_regiao': 0
        })
        mov_counter += 1

    # Registrar fluxo diário (Saídas por Consumo e Desperdícios Ocasionais)
    mapa_lote_produto = df_lotes.groupby('produto_id')['lote_id'].first().to_dict()

    for i in range(365):
        data_atual = data_inicio + timedelta(days=i)
        data_str = data_atual.strftime('%Y-%m-%d')
        dia_num = data_atual.weekday()
        dia_mes = data_atual.day
        mes = data_atual.month
        nome_dia = dias_semana_nome[dia_num]

        eh_feriado = 1 if (dia_mes in [1, 12, 25] or (mes == 9 and dia_mes == 7)) else 0
        fator_pagamento = 1.20 if dia_mes <= 10 else 1.0

        if mes in [5, 6, 7, 8]:
            clima = np.random.choice(climas_possiveis, p=[0.3, 0.2, 0.2, 0.3])
            temp = float(np.random.uniform(12.0, 22.0))
        else:
            clima = np.random.choice(climas_possiveis, p=[0.6, 0.2, 0.1, 0.1])
            temp = float(np.random.uniform(22.0, 35.0))

        evento_regiao = 1 if (dia_num in [4, 5] and np.random.rand() > 0.6) else 0

        for _, prod in df_produtos_config.iterrows():
            p_id = prod['produto_id']
            lote_id_associado = mapa_lote_produto.get(p_id, 'L001')
            
            base = prod['base_fds'] if dia_num in [4, 5, 6] else prod['base_util']
            mult = fator_pagamento

            if eh_feriado: mult *= 1.30
            if evento_regiao: mult *= 1.20

            cat = prod['categoria']
            if cat == 'Bebidas':
                if temp > 28.0: mult *= 1.45
                if clima in ['Chuvoso', 'Frio']: mult *= 0.50
            elif cat == 'Hortifrúti':
                if temp > 28.0: mult *= 1.25
                if clima == 'Frio': mult *= 0.65
            elif cat == 'Carnes' and clima == 'Chuvoso':
                mult *= 0.85

            qtd_saida = round(max(0.5, (base * mult) + np.random.normal(0, base * 0.05)), 2)

            # 1. Movimentação de SAIDA_CONSUMO
            movimentacoes_data.append({
                'movimentacao_id': f'M{mov_counter}',
                'lote_id': lote_id_associado,
                'produto_id': p_id,
                'tipo_movimentacao': 'SAIDA_CONSUMO',
                'quantidade': qtd_saida,
                'motivo_desperdicio': None,
                'usuario_id': 2, # Operador da Cozinha
                'data_hora': f"{data_str} 21:30:00",
                'dia_semana': nome_dia,
                'eh_feriado': eh_feriado,
                'clima': clima,
                'temp_media_c': round(temp, 1),
                'evento_regiao': evento_regiao
            })
            mov_counter += 1

            # 2. Movimentação de DESPERDICIO (Ocorre ocasionalmente em ~5% dos dias)
            if np.random.rand() < 0.05:
                qtd_desperdicio = round(np.random.uniform(0.2, 1.5), 2)
                motivo = np.random.choice(motivos_desperdicio)

                movimentacoes_data.append({
                    'movimentacao_id': f'M{mov_counter}',
                    'lote_id': lote_id_associado,
                    'produto_id': p_id,
                    'tipo_movimentacao': 'DESPERDICIO',
                    'quantidade': qtd_desperdicio,
                    'motivo_desperdicio': motivo,
                    'usuario_id': 2,
                    'data_hora': f"{data_str} 22:00:00",
                    'dia_semana': nome_dia,
                    'eh_feriado': eh_feriado,
                    'clima': clima,
                    'temp_media_c': round(temp, 1),
                    'evento_regiao': evento_regiao
                })
                mov_counter += 1

    df_mov = pd.DataFrame(movimentacoes_data)
    df_mov.to_csv('movimentacoes_estoque.csv', index=False, encoding='utf-8-sig')
    print(f"✔ [3/3] 'movimentacoes_estoque.csv' gerado com {len(df_mov)} registros de movimentação.")
    print("\n Base de dados unificada e atualizada com sucesso!")

if __name__ == "__main__":
    gerar_massa_testes_restaurante()