import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def criar_base_completa():
    print("Gerando arquivos CSV para o sistema do restaurante...")

    # 1. PRODUTOS.CSV (RF01, RF02, RF06)
    df_produtos = pd.DataFrame([
        {'produto_id': 101, 'nome': 'Contra-Filé', 'categoria': 'Carnes', 'unidade_medida': 'kg', 'estoque_minimo': 15.0},
        {'produto_id': 102, 'nome': 'Queijo Mozarela', 'categoria': 'Laticínios', 'unidade_medida': 'kg', 'estoque_minimo': 5.0},
        {'produto_id': 103, 'nome': 'Arroz Arbóreo', 'categoria': 'Grãos', 'unidade_medida': 'kg', 'estoque_minimo': 4.0},
        {'produto_id': 104, 'nome': 'Alface Crespa', 'categoria': 'Hortifrúti', 'unidade_medida': 'Un', 'estoque_minimo': 10.0}
    ])
    df_produtos.to_csv('produtos.csv', index=False, encoding='utf-8-sig')

    # 2. LOTES_ESTOQUE.CSV (RF02, RF03 - Modelo FIFO, RF04)
    df_lotes = pd.DataFrame([
        {'lote_id': 'L01', 'produto_id': 101, 'codigo_lote': 'LOT-20260910-A', 'quantidade_inicial': 50.0, 'quantidade_atual': 12.5, 'data_validade': '2026-09-28', 'data_entrada': '2026-09-10'},
        {'lote_id': 'L02', 'produto_id': 101, 'codigo_lote': 'LOT-20260918-B', 'quantidade_inicial': 40.0, 'quantidade_atual': 40.0, 'data_validade': '2026-10-05', 'data_entrada': '2026-09-18'},
        {'lote_id': 'L03', 'produto_id': 102, 'codigo_lote': 'LOT-20260911-A', 'quantidade_inicial': 15.0, 'quantidade_atual': 6.7, 'data_validade': '2026-09-30', 'data_entrada': '2026-09-11'}
    ])
    df_lotes.to_csv('lotes_estoque.csv', index=False, encoding='utf-8-sig')

    # 3. MOVIMENTACOES_ESTOQUE.CSV (RF03, RF04, RF07, RF08)
    df_movimentacoes = pd.DataFrame([
        {'movimentacao_id': 'M1001', 'lote_id': 'L01', 'tipo_movimentacao': 'ENTRADA', 'quantidade': 50.0, 'motivo_desperdicio': None, 'usuario_id': 1, 'data_hora': '2026-09-10 07:30'},
        {'movimentacao_id': 'M1002', 'lote_id': 'L01', 'tipo_movimentacao': 'SAIDA_CONSUMO', 'quantidade': 18.5, 'motivo_desperdicio': None, 'usuario_id': 2, 'data_hora': '2026-09-11 22:30'},
        {'movimentacao_id': 'M1003', 'lote_id': 'L01', 'tipo_movimentacao': 'DESPERDICIO', 'quantidade': 1.2, 'motivo_desperdicio': 'Validade / Avaria', 'usuario_id': 2, 'data_hora': '2026-09-12 11:00'}
    ])
    df_movimentacoes.to_csv('movimentacoes_estoque.csv', index=False, encoding='utf-8-sig')

    # 4. HISTORICO_VENDAS_DIARIO.CSV (RF09, RF10, RF13)
    np.random.seed(42)
    data_inicio = datetime(2026, 6, 1)
    dias_semana_nome = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    climas = ['Ensolarado', 'Chuvoso', 'Nublado', 'Frio']
    dados_vendas = []

    for i in range(90):
        data_atual = data_inicio + timedelta(days=i)
        dia_num = data_atual.weekday()
        eh_feriado = 1 if (i % 15 == 0) else 0
        clima = np.random.choice(climas, p=[0.4, 0.2, 0.2, 0.2])
        temp = float(np.random.uniform(15.0, 32.0))
        evento = 1 if (dia_num in [4, 5] and np.random.rand() > 0.5) else 0
        
        # Padrão de demanda: fins de semana vendem mais
        base_demanda = 25.0 if dia_num in [4, 5, 6] else 12.0
        if eh_feriado: base_demanda += 10.0
        if evento: base_demanda += 8.0
        if clima == 'Chuvoso': base_demanda -= 3.0
        qtd = round(max(5.0, base_demanda + np.random.normal(0, 3)), 2)

        dados_vendas.append({
            'venda_id': f'V{500+i}', 'data': data_atual.strftime('%Y-%m-%d'), 'produto_id': 101,
            'qtd_vendida': qtd, 'dia_semana': dias_semana_nome[dia_num], 'eh_feriado': eh_feriado,
            'clima': clima, 'temp_media_c': round(temp, 1), 'evento_regiao': evento
        })
    
    df_vendas = pd.DataFrame(dados_vendas)
    df_vendas.to_csv('historico_vendas_diario.csv', index=False, encoding='utf-8-sig')

    # 5. USUARIOS.CSV (RF15)
    df_usuarios = pd.DataFrame([
        {'usuario_id': 1, 'nome': 'Carlos Eduardo', 'email': 'carlos@restaurante.com', 'nivel_acesso': 'Administrador'},
        {'usuario_id': 2, 'nome': 'Ana Souza', 'email': 'ana.cozinha@restaurante.com', 'nivel_acesso': 'Operador_Cozinha'},
        {'usuario_id': 3, 'nome': 'Roberto Lima', 'email': 'roberto.compras@restaurante.com', 'nivel_acesso': 'Gestor_Compras'}
    ])
    df_usuarios.to_csv('usuarios.csv', index=False, encoding='utf-8-sig')

    print("✔ Sucesso! Todos os arquivos CSV de suporte foram gerados na pasta do seu projeto.")

if __name__ == "__main__":
    criar_base_completa()