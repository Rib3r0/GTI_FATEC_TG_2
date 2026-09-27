import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gestão de Estoque & IA",
    page_icon="📦",
    layout="wide"
)

DB_NAME = "gestao_estoque.db"

# -----------------------------------------------------------------------------
# BANCO DE DADOS & INICIALIZAÇÃO
# -----------------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # RF01, RF02, RF06 - Produtos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS produtos (
        produto_id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        categoria TEXT NOT NULL,
        unidade_medida TEXT NOT NULL,
        estoque_minimo REAL NOT NULL
    )''')
    
    # RF03, RF04 - Lotes (FIFO / PEPS)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lotes_estoque (
        lote_id INTEGER PRIMARY KEY AUTOINCREMENT,
        produto_id INTEGER,
        codigo_lote TEXT NOT NULL,
        quantidade_inicial REAL NOT NULL,
        quantidade_atual REAL NOT NULL,
        data_validade DATE NOT NULL,
        data_entrada DATETIME NOT NULL,
        FOREIGN KEY (produto_id) REFERENCES produtos (produto_id)
    )''')
    
    # RF03, RF04, RF07, RF08 - Movimentações
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS movimentacoes_estoque (
        movimentacao_id INTEGER PRIMARY KEY AUTOINCREMENT,
        lote_id INTEGER,
        tipo_movimentacao TEXT NOT NULL,
        quantidade REAL NOT NULL,
        motivo_desperdicio TEXT,
        usuario TEXT,
        data_hora DATETIME NOT NULL,
        FOREIGN KEY (lote_id) REFERENCES lotes_estoque (lote_id)
    )''')
    
    # RF09, RF10, RF13 - Histórico para IA
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historico_movimentacoes (
        historico_id INTEGER PRIMARY KEY AUTOINCREMENT,
        produto_id INTEGER,
        data DATE NOT NULL,
        quantidade_saida REAL NOT NULL,
        dia_semana INTEGER,
        eh_fim_semana INTEGER,
        FOREIGN KEY (produto_id) REFERENCES produtos (produto_id)
    )''')
    
    # RF15 - Usuários
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS usuarios (
        usuario_id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        nivel_acesso TEXT NOT NULL
    )''')
    
    conn.commit()
    
    # Semeadura de Dados Iniciais caso esteja vazio
    cursor.execute("SELECT COUNT(*) FROM produtos")
    if cursor.fetchone()[0] == 0:
        produtos_iniciais = [
            ("Contra-Filé", "Carnes", "kg", 15.0),
            ("Queijo Mozarela", "Laticínios", "kg", 5.0),
            ("Arroz Arbóreo", "Grãos", "kg", 10.0),
            ("Batata Frita Congelada", "Congelados", "kg", 12.0),
            ("Refrigerante Cola 2L", "Bebidas", "Un", 20.0)
        ]
        cursor.executemany("INSERT INTO produtos (nome, categoria, unidade_medida, estoque_minimo) VALUES (?, ?, ?, ?)", produtos_iniciais)
        conn.commit()
        
        # Inserção de Lotes e Histórico Fictício de 90 dias para a IA
        hoje = datetime.now()
        for p_id in range(1, 6):
            # Lote Inicial
            cursor.execute('''
            INSERT INTO lotes_estoque (produto_id, codigo_lote, quantidade_inicial, quantidade_atual, data_validade, data_entrada)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (p_id, f"LOT-{p_id}-001", 50.0, 25.0, (hoje + timedelta(days=30)).strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d %H:%M:%S")))
            
            # Histórico Diário de Saídas
            for d in range(90, 0, -1):
                data_hist = hoje - timedelta(days=d)
                is_weekend = 1 if data_hist.weekday() >= 5 else 0
                qtd_saida = np.random.uniform(3.0, 8.0) * (1.6 if is_weekend else 1.0)
                cursor.execute('''
                INSERT INTO historico_movimentacoes (produto_id, data, quantidade_saida, dia_semana, eh_fim_semana)
                VALUES (?, ?, ?, ?, ?)
                ''', (p_id, data_hist.strftime("%Y-%m-%d"), round(qtd_saida, 2), data_hist.weekday(), is_weekend))
        
        # Usuários Padrão
        usuarios_iniciais = [
            ("Administrador", "admin@restaurante.com", "Administrador"),
            ("Operador Estoque", "operador@restaurante.com", "Operador"),
            ("Gestor Compras", "compras@restaurante.com", "Gestor_Compras")
        ]
        cursor.executemany("INSERT INTO usuarios (nome, email, nivel_acesso) VALUES (?, ?, ?)", usuarios_iniciais)
        conn.commit()
        
    conn.close()

init_db()

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE ESTOQUE E FIFO
# -----------------------------------------------------------------------------
def obter_estoque_atual(produto_id):
    conn = get_db_connection()
    res = conn.execute("SELECT SUM(quantidade_atual) FROM lotes_estoque WHERE produto_id = ?", (produto_id,)).fetchone()[0]
    conn.close()
    return res if res else 0.0

def registrar_saida_fifo(produto_id, qtd_solicitada, tipo_mov, motivo, usuario):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Buscar lotes do produto ordenados por data de validade (FIFO / PEPS)
    lotes = cursor.execute('''
        SELECT lote_id, quantidade_atual FROM lotes_estoque 
        WHERE produto_id = ? AND quantidade_atual > 0 
        ORDER BY data_validade ASC
    ''', (produto_id,)).fetchall()
    
    qtd_restante = qtd_solicitada
    for lote in lotes:
        if qtd_restante <= 0:
            break
        
        lote_id = lote['lote_id']
        qtd_disponivel = lote['quantidade_atual']
        
        if qtd_disponivel >= qtd_restante:
            nova_qtd = qtd_disponivel - qtd_restante
            cursor.execute("UPDATE lotes_estoque SET quantidade_atual = ? WHERE lote_id = ?", (nova_qtd, lote_id))
            cursor.execute('''
                INSERT INTO movimentacoes_estoque (lote_id, tipo_movimentacao, quantidade, motivo_desperdicio, usuario, data_hora)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lote_id, tipo_mov, qtd_restante, motivo, usuario, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            qtd_restante = 0
        else:
            cursor.execute("UPDATE lotes_estoque SET quantidade_atual = 0 WHERE lote_id = ?", (lote_id,))
            cursor.execute('''
                INSERT INTO movimentacoes_estoque (lote_id, tipo_movimentacao, quantidade, motivo_desperdicio, usuario, data_hora)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (lote_id, tipo_mov, qtd_disponivel, motivo, usuario, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            qtd_restante -= qtd_disponivel
            
    conn.commit()
    conn.close()
    return qtd_solicitada - qtd_restante

# -----------------------------------------------------------------------------
# NAVEGAÇÃO LATERAL E PERMISSÕES (RF15)
# -----------------------------------------------------------------------------
st.sidebar.title("🍱 Gestão de Estoque & IA")
usuario_ativo = st.sidebar.selectbox("Usuário Ativo (RF15):", ["Administrador", "Operador Estoque", "Gestor Compras"])

menu = st.sidebar.radio("Navegação", [
    "📊 Painel Principal",
    "📦 Cadastro de Produtos (RF01/02)",
    "🔄 Movimentação & FIFO (RF03/04/07)",
    "🤖 Previsão de Demanda & Compras (RF09-13)",
    "📈 Relatórios (RF14)"
])

# -----------------------------------------------------------------------------
# 1. PAINEL PRINCIPAL (RF05, RF06)
# -----------------------------------------------------------------------------
if menu == "📊 Painel Principal":
    st.header("📊 Visão Geral do Estoque e Alertas")
    
    conn = get_db_connection()
    df_prods = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    
    df_prods['estoque_atual'] = df_prods['produto_id'].apply(obter_estoque_atual)
    df_prods['status_alerta'] = df_prods['estoque_atual'] < df_prods['estoque_minimo']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Produtos", len(df_prods))
    col2.metric("Itens Críticos (Abaixo do Mínimo)", int(df_prods['status_alerta'].sum()))
    col3.metric("Status do Sistema", "Operacional", delta="100% Uptime")
    
    st.subheader("⚠️ Alertas de Estoque Mínimo (RF06)")
    alertas = df_prods[df_prods['status_alerta']]
    if not alertas.empty:
        for _, row in alertas.iterrows():
            st.error(f"**Atenção:** O produto **{row['nome']}** está com {row['estoque_atual']} {row['unidade_medida']} (Mínimo exigido: {row['estoque_minimo']} {row['unidade_medida']}).")
    else:
        st.success("Todos os produtos estão com níveis de estoque acima do limite mínimo!")
        
    st.subheader("📦 Consulta Geral do Estoque (RF05)")
    st.dataframe(df_prods[['produto_id', 'nome', 'categoria', 'estoque_atual', 'unidade_medida', 'estoque_minimo']], use_container_width=True)

# -----------------------------------------------------------------------------
# 2. CADASTRO DE PRODUTOS (RF01, RF02)
# -----------------------------------------------------------------------------
elif menu == "📦 Cadastro de Produtos (RF01/02)":
    st.header("📦 Gestão do Cadastro de Produtos")
    
    with st.form("form_produto"):
        st.subheader("Cadastrar Novo Produto")
        nome = st.text_input("Nome do Produto")
        categoria = st.selectbox("Categoria", ["Carnes", "Grãos", "Laticínios", "Hortifrúti", "Congelados", "Bebidas", "Mercearia"])
        unidade = st.selectbox("Unidade de Medida", ["kg", "L", "Un"])
        est_min = st.number_input("Estoque Mínimo de Segurança", min_value=0.0, value=10.0, step=1.0)
        
        btn_salvar = st.form_submit_button("Salvar Produto")
        if btn_salvar and nome:
            conn = get_db_connection()
            conn.execute("INSERT INTO produtos (nome, categoria, unidade_medida, estoque_minimo) VALUES (?, ?, ?, ?)",
                         (nome, categoria, unidade, est_min))
            conn.commit()
            conn.close()
            st.success(f"Produto '{nome}' cadastrado com sucesso!")
            st.rerun()

    st.subheader("Produtos Cadastrados")
    conn = get_db_connection()
    df_p = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    st.dataframe(df_p, use_container_width=True)

# -----------------------------------------------------------------------------
# 3. MOVIMENTAÇÃO & FIFO (RF03, RF04, RF07, RF08)
# -----------------------------------------------------------------------------
elif menu == "🔄 Movimentação & FIFO (RF03/04/07)":
    st.header("🔄 Movimentação de Estoque e Registro de Lotes")
    
    conn = get_db_connection()
    produtos = conn.execute("SELECT produto_id, nome, unidade_medida FROM produtos").fetchall()
    conn.close()
    
    dict_prods = {p['nome']: (p['produto_id'], p['unidade_medida']) for p in produtos}
    
    aba1, aba2 = st.tabs(["📥 Entrada de Novo Lote", "📤 Saída de Consumo / Desperdício (FIFO)"])
    
    with aba1:
        st.subheader("Registrar Entrada de Lote (Compra)")
        prod_sel = st.selectbox("Selecione o Produto para Entrada", list(dict_prods.keys()), key="ent_prod")
        cod_lote = st.text_input("Código do Lote / Nota Fiscal", f"LOT-{datetime.now().strftime('%Y%m%d%H%M')}")
        qtd_entrada = st.number_input("Quantidade Recebida", min_value=0.1, value=10.0)
        val_lote = st.date_input("Data de Validade", datetime.now() + timedelta(days=30))
        
        if st.button("Registrar Entrada de Lote"):
            p_id, _ = dict_prods[prod_sel]
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO lotes_estoque (produto_id, codigo_lote, quantidade_inicial, quantidade_atual, data_validade, data_entrada)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (p_id, cod_lote, qtd_entrada, qtd_entrada, val_lote.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            lote_id = cursor.lastrowid
            cursor.execute('''
                INSERT INTO movimentacoes_estoque (lote_id, tipo_movimentacao, quantidade, motivo_desperdicio, usuario, data_hora)
                VALUES (?, 'ENTRADA', ?, NULL, ?, ?)
            ''', (lote_id, qtd_entrada, usuario_ativo, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            conn.commit()
            conn.close()
            st.success(f"Lote {cod_lote} adicionado ao estoque!")
            st.rerun()

    with aba2:
        st.subheader("Registrar Saída (Baixa Automática no Modelo FIFO / PEPS)")
        prod_sai = st.selectbox("Selecione o Produto para Saída", list(dict_prods.keys()), key="sai_prod")
        p_id, un = dict_prods[prod_sai]
        est_disponivel = obter_estoque_atual(p_id)
        
        st.info(f"Estoque Disponível Atual: **{est_disponivel} {un}**")
        
        tipo_saida = st.radio("Tipo de Saída", ["SAIDA_CONSUMO", "DESPERDICIO"])
        qtd_saida = st.number_input(f"Quantidade a dar baixa ({un})", min_value=0.1, max_value=float(est_disponivel) if est_disponivel > 0 else 1.0, value=1.0)
        
        motivo = ""
        if tipo_saida == "DESPERDICIO":
            motivo = st.selectbox("Motivo do Desperdício (RF07)", ["Validade Vencida", "Avaria / Quebra", "Erro de Preparações", "Outros"])
            
        if st.button("Confirmar Baixa de Estoque"):
            if est_disponivel >= qtd_saida:
                atendido = registrar_saida_fifo(p_id, qtd_saida, tipo_saida, motivo, usuario_ativo)
                
                # Atualiza também o histórico diário para alimentar a IA
                conn = get_db_connection()
                hoje_str = datetime.now().strftime("%Y-%m-%d")
                conn.execute('''
                    INSERT INTO historico_movimentacoes (produto_id, data, quantidade_saida, dia_semana, eh_fim_semana)
                    VALUES (?, ?, ?, ?, ?)
                ''', (p_id, hoje_str, qtd_saida, datetime.now().weekday(), 1 if datetime.now().weekday() >= 5 else 0))
                conn.commit()
                conn.close()
                
                st.success(f"Baixa de {atendido} {un} realizada utilizando a regra do lote mais antigo primeiro (FIFO)!")
                st.rerun()
            else:
                st.error("Quantidade solicitada maior que o estoque total disponível.")

# -----------------------------------------------------------------------------
# 4. PREVISÃO DE DEMANDA & INTELIGÊNCIA ARTIFICIAL (RF09 AO RF13)
# -----------------------------------------------------------------------------
elif menu == "🤖 Previsão de Demanda & Compras (RF09-13)":
    st.header("🤖 Previsão de Demanda e Sugestão de Compras com IA")
    st.markdown("O sistema analisa o histórico de movimentações, dias da semana e sazonalidades para calcular a compra exata necessária (RF10, RF13).")
    
    conn = get_db_connection()
    produtos = conn.execute("SELECT produto_id, nome, unidade_medida, estoque_minimo FROM produtos").fetchall()
    
    dias_previsao = st.slider("Selecione o horizonte de previsão futuro (RF11):", min_value=1, max_value=30, value=7, help="Calcula a previsão para N dias futuros")
    
    previsoes_geradas = []
    
    for prod in produtos:
        p_id = prod['produto_id']
        nome_prod = prod['nome']
        un = prod['unidade_medida']
        est_min = prod['estoque_minimo']
        
        # Coleta de Dados para o Modelo de Machine Learning (RF09)
        df_hist = pd.read_sql_query("SELECT quantidade_saida, dia_semana, eh_fim_semana FROM historico_movimentacoes WHERE produto_id = ?", conn, params=(p_id,))
        
        if len(df_hist) >= 10:
            X = df_hist[['dia_semana', 'eh_fim_semana']]
            y = df_hist['quantidade_saida']
            
            # Treinamento do Modelo RandomForestRegressor (RF10)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Gerando os cenários para os próximos N dias
            hoje = datetime.now()
            datas_futuras = [hoje + timedelta(days=i) for i in range(1, dias_previsao + 1)]
            df_futuro = pd.DataFrame({
                'dia_semana': [d.weekday() for d in datas_futuras],
                'eh_fim_semana': [1 if d.weekday() >= 5 else 0 for d in datas_futuras]
            })
            
            # Previsão das demandas diárias
            demandas_diarias = model.predict(df_futuro)
            demanda_total_prevista = np.sum(demandas_diarias)
            
            estoque_atual = obter_estoque_atual(p_id)
            
            # Cálculo da Sugestão de Compras (RF12)
            # Fórmulas: (Demanda Prevista + Estoque Mínimo) - Estoque Atual
            sugestao_compra = max(0.0, (demanda_total_prevista + est_min) - estoque_atual)
            
            previsoes_geradas.append({
                "Produto": nome_prod,
                "Estoque Atual": f"{estoque_atual:.2f} {un}",
                "Estoque Mínimo": f"{est_min:.2f} {un}",
                "Demanda Prevista (Próx. Período)": f"{demanda_total_prevista:.2f} {un}",
                "Sugestão de Compra": f"{sugestao_compra:.2f} {un}" if sugestao_compra > 0 else "Estoque Suficiente"
            })
    
    conn.close()
    
    st.subheader(f"📊 Relatório de Previsões para os próximos {dias_previsao} dias (RF11, RF12, RNF12)")
    if previsoes_geradas:
        df_res = pd.DataFrame(previsoes_geradas)
        st.dataframe(df_res, use_container_width=True)
        
        st.info("💡 **Entendimento sem jargões técnicos (RNF12):** Se a 'Sugestão de Compra' for maior que zero, é recomendável realizar o pedido ao fornecedor imediatamente para cobrir as vendas previstas sem risco de zerar o estoque mínimo de segurança.")
    else:
        st.warning("Dados históricos insuficientes para treinar o modelo de Inteligência Artificial.")

# -----------------------------------------------------------------------------
# 5. RELATÓRIOS (RF14)
# -----------------------------------------------------------------------------
elif menu == "📈 Relatórios (RF14)":
    st.header("📈 Relatórios de Desperdícios, Movimentações e Estoque")
    
    conn = get_db_connection()
    
    aba_rel1, aba_rel2 = st.tabs(["📋 Histórico Completo de Movimentações", "🗑️ Relatório de Desperdícios"])
    
    with aba_rel1:
        st.subheader("Todas as Movimentações Registradas (RF08, RF14)")
        df_movs = pd.read_sql_query('''
            SELECT m.movimentacao_id, p.nome as produto, m.tipo_movimentacao, m.quantidade, 
                   p.unidade_medida, m.motivo_desperdicio, m.usuario, m.data_hora
            FROM movimentacoes_estoque m
            JOIN lotes_estoque l ON m.lote_id = l.lote_id
            JOIN produtos p ON l.produto_id = p.produto_id
            ORDER BY m.data_hora DESC
        ''', conn)
        st.dataframe(df_movs, use_container_width=True)
        
    with aba_rel2:
        st.subheader("Análise de Perdas e Desperdícios (RF07, RF14)")
        df_desp = pd.read_sql_query('''
            SELECT p.nome as produto, SUM(m.quantidade) as total_desperdicado, m.motivo_desperdicio
            FROM movimentacoes_estoque m
            JOIN lotes_estoque l ON m.lote_id = l.lote_id
            JOIN produtos p ON l.produto_id = p.produto_id
            WHERE m.tipo_movimentacao = 'DESPERDICIO'
            GROUP BY p.nome, m.motivo_desperdicio
        ''', conn)
        
        if not df_desp.empty:
            fig = px.bar(df_desp, x="produto", y="total_desperdicado", color="motivo_desperdicio", 
                         title="Volume de Desperdício por Produto e Motivo", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_desp, use_container_width=True)
        else:
            st.success("Nenhum registro de desperdício até o momento!")
            
    conn.close()