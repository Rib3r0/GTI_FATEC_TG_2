import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO E ESTILIZAÇÃO VISUAL (UI/UX)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gestão de Estoque & IA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeção de CSS personalizado para polimento do layout
st.markdown("""
<style>
    /* Estilização de Cards e Métricas */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-card-danger {
        background-color: #fff5f5;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #dc3545;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    /* Estilização de Cabeçalhos de Seções */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

DB_NAME = "gestao_estoque.db"

# -----------------------------------------------------------------------------
# INICIALIZAÇÃO DO BANCO DE DADOS
# -----------------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS produtos (
        produto_id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        categoria TEXT NOT NULL,
        unidade_medida TEXT NOT NULL,
        estoque_minimo REAL NOT NULL
    )''')
    
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
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS usuarios (
        usuario_id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        nivel_acesso TEXT NOT NULL
    )''')
    
    conn.commit()
    
    # Injeção de dados de teste (caso o banco esteja zerado)
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
        
        hoje = datetime.now()
        for p_id in range(1, 6):
            cursor.execute('''
            INSERT INTO lotes_estoque (produto_id, codigo_lote, quantidade_inicial, quantidade_atual, data_validade, data_entrada)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (p_id, f"LOT-{p_id}-001", 50.0, 14.0 if p_id in [1, 4] else 35.0, (hoje + timedelta(days=30)).strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d %H:%M:%S")))
            
            for d in range(90, 0, -1):
                data_hist = hoje - timedelta(days=d)
                is_weekend = 1 if data_hist.weekday() >= 5 else 0
                qtd_saida = np.random.uniform(3.0, 8.0) * (1.6 if is_weekend else 1.0)
                cursor.execute('''
                INSERT INTO historico_movimentacoes (produto_id, data, quantidade_saida, dia_semana, eh_fim_semana)
                VALUES (?, ?, ?, ?, ?)
                ''', (p_id, data_hist.strftime("%Y-%m-%d"), round(qtd_saida, 2), data_hist.weekday(), is_weekend))
        
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
# FUNÇÕES DE NEGÓCIO E LÓGICA DE ESTOQUE (FIFO)
# -----------------------------------------------------------------------------
def obter_estoque_atual(produto_id):
    conn = get_db_connection()
    res = conn.execute("SELECT SUM(quantidade_atual) FROM lotes_estoque WHERE produto_id = ?", (produto_id,)).fetchone()[0]
    conn.close()
    return res if res else 0.0

def registrar_saida_fifo(produto_id, qtd_solicitada, tipo_mov, motivo, usuario):
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
# NAVEGAÇÃO LATERAL (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1532/1532688.png", width=70)
st.sidebar.title("Gestão & IA")
usuario_ativo = st.sidebar.selectbox("👤 Perfil Conectado (RF15):", ["Administrador", "Operador Estoque", "Gestor Compras"])

st.sidebar.divider()
menu = st.sidebar.radio("📌 Menu de Navegação", [
    "📊 Painel de Controle (Dashboard)",
    "📦 Gestão de Produtos",
    "🔄 Movimentação & Regra FIFO",
    "🤖 Previsão de IA & Compras",
    "📈 Relatórios & Auditoria"
])

# -----------------------------------------------------------------------------
# 1. PAINEL DE CONTROLE / DASHBOARD (RF05, RF06)
# -----------------------------------------------------------------------------
if menu == "📊 Painel de Controle (Dashboard)":
    st.title("📊 Painel Geral de Estoque e Alertas")
    st.caption("Acompanhamento de saldo, alertas automáticos e comparativos de estoque em tempo real.")
    
    conn = get_db_connection()
    df_prods = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    
    df_prods['estoque_atual'] = df_prods['produto_id'].apply(obter_estoque_atual)
    df_prods['status'] = np.where(df_prods['estoque_atual'] < df_prods['estoque_minimo'], "⚠️ Baixo", "✅ Ok")
    
    total_itens = len(df_prods)
    itens_criticos = len(df_prods[df_prods['estoque_atual'] < df_prods['estoque_minimo']])
    total_unidades = df_prods['estoque_atual'].sum()
    
    # Cards de KPIs no topo
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Produtos Cadastrados", total_itens, help="Insumos ativos no sistema")
    c2.metric("Itens Críticos (Abaixo do Mínimo)", itens_criticos, delta_color="inverse", delta=f"{itens_criticos} alerta(s)" if itens_criticos > 0 else "Estoque Seguro")
    c3.metric("Volume Total no Estoque", f"{total_unidades:.1f} Un/Kg", help="Soma de todos os saldos de lotes")
    
    st.divider()
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("<div class='section-header'>📊 Comparativo: Estoque Atual vs Mínimo Exigido</div>", unsafe_allow_html=True)
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            x=df_prods['nome'], y=df_prods['estoque_atual'],
            name='Estoque Atual', marker_color='#007bff'
        ))
        fig_bars.add_trace(go.Bar(
            x=df_prods['nome'], y=df_prods['estoque_minimo'],
            name='Estoque Mínimo de Segurança', marker_color='#dc3545'
        ))
        fig_bars.update_layout(barmode='group', template='plotly_white', margin=dict(l=20, r=20, t=30, b=20), height=350)
        st.plotly_chart(fig_bars, use_container_width=True)
        
    with col_right:
        st.markdown("<div class='section-header'>🚨 Alertas Críticos (RF06)</div>", unsafe_allow_html=True)
        alertas = df_prods[df_prods['estoque_atual'] < df_prods['estoque_minimo']]
        if not alertas.empty:
            for _, row in alertas.iterrows():
                st.error(f"**{row['nome']}**: Apenas {row['estoque_atual']} {row['unidade_medida']} em estoque (Mínimo: {row['estoque_minimo']} {row['unidade_medida']})")
        else:
            st.success("Nenhum produto está abaixo do limite de segurança no momento.")

    st.markdown("<div class='section-header'>📋 Visão Tabular do Saldo de Produtos (RF05)</div>", unsafe_allow_html=True)
    st.dataframe(
        df_prods[['produto_id', 'nome', 'categoria', 'estoque_atual', 'estoque_minimo', 'unidade_medida', 'status']],
        column_config={
            "produto_id": "ID",
            "nome": "Produto",
            "categoria": "Categoria",
            "estoque_atual": st.column_config.ProgressColumn(
                "Nível de Estoque",
                format="%.1f",
                min_value=0,
                max_value=float(df_prods['estoque_atual'].max() * 1.2) if len(df_prods) > 0 else 100
            ),
            "estoque_minimo": "Mínimo",
            "unidade_medida": "Unidade",
            "status": "Status"
        },
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------------------------------------------------
# 2. GESTÃO DE PRODUTOS (RF01, RF02)
# -----------------------------------------------------------------------------
elif menu == "📦 Gestão de Produtos":
    st.title("📦 Cadastro e Gestão de Produtos")
    st.caption("Cadastre novos insumos, defina unidades de medida e configure estoques de segurança.")
    
    with st.expander("➕ Cadastrar Novo Produto", expanded=False):
        with st.form("form_novo_produto", clear_on_submit=True):
            f1, f2 = st.columns(2)
            nome = f1.text_input("Nome do Produto / Insumo")
            categoria = f2.selectbox("Categoria", ["Carnes", "Grãos", "Laticínios", "Hortifrúti", "Congelados", "Bebidas", "Mercearia"])
            
            f3, f4 = st.columns(2)
            unidade = f3.selectbox("Unidade de Medida", ["kg", "L", "Un"])
            est_min = f4.number_input("Estoque Mínimo de Segurança", min_value=0.0, value=10.0, step=1.0)
            
            btn = st.form_submit_button("Salvar Registro")
            if btn and nome:
                conn = get_db_connection()
                conn.execute("INSERT INTO produtos (nome, categoria, unidade_medida, estoque_minimo) VALUES (?, ?, ?, ?)",
                             (nome, categoria, unidade, est_min))
                conn.commit()
                conn.close()
                st.success(f"Produto '{nome}' cadastrado com sucesso!")
                st.rerun()

    st.markdown("<div class='section-header'>Produtos Cadastrados no Sistema</div>", unsafe_allow_html=True)
    conn = get_db_connection()
    df_p = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    st.dataframe(df_p, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# 3. MOVIMENTAÇÃO & FIFO (RF03, RF04, RF07, RF08)
# -----------------------------------------------------------------------------
elif menu == "🔄 Movimentação & Regra FIFO":
    st.title("🔄 Movimentação de Estoque")
    st.caption("Entradas de lotes e baixas de estoque calculadas automaticamente pelo lote mais antigo (FIFO/PEPS).")
    
    conn = get_db_connection()
    produtos = conn.execute("SELECT produto_id, nome, unidade_medida FROM produtos").fetchall()
    conn.close()
    
    dict_prods = {p['nome']: (p['produto_id'], p['unidade_medida']) for p in produtos}
    
    t1, t2 = st.tabs(["📥 Nova Entrada (Lote)", "📤 Baixa / Consumo / Desperdício"])
    
    with t1:
        st.subheader("Registrar Entrada de Mercadoria")
        c1, c2 = st.columns(2)
        prod_sel = c1.selectbox("Selecione o Produto", list(dict_prods.keys()), key="ent_prod")
        cod_lote = c2.text_input("Código do Lote / NF", f"LOT-{datetime.now().strftime('%Y%m%d%H%M')}")
        
        c3, c4 = st.columns(2)
        qtd_entrada = c3.number_input("Quantidade Recebida", min_value=0.1, value=10.0)
        val_lote = c4.date_input("Data de Validade", datetime.now() + timedelta(days=30))
        
        if st.button("📥 Salvar Entrada de Lote", type="primary"):
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
            st.success(f"Entrada de {qtd_entrada} do lote {cod_lote} gravada com sucesso!")
            st.rerun()

    with t2:
        st.subheader("Registrar Saída (Baixa Automática por FIFO)")
        prod_sai = st.selectbox("Selecione o Produto", list(dict_prods.keys()), key="sai_prod")
        p_id, un = dict_prods[prod_sai]
        est_disponivel = obter_estoque_atual(p_id)
        
        st.info(f"💡 Saldo total disponível em lotes ativos: **{est_disponivel:.2f} {un}**")
        
        c1, c2 = st.columns(2)
        tipo_saida = c1.radio("Finalidade da Saída", ["SAIDA_CONSUMO", "DESPERDICIO"])
        qtd_saida = c2.number_input(f"Quantidade a dar baixa ({un})", min_value=0.1, max_value=float(est_disponivel) if est_disponivel > 0 else 1.0, value=1.0)
        
        motivo = None
        if tipo_saida == "DESPERDICIO":
            motivo = st.selectbox("Motivo da Perda (RF07)", ["Validade Vencida", "Avaria / Quebra", "Erro de Preparação", "Outros"])
            
        if st.button("📤 Confirmar Baixa de Estoque", type="primary"):
            if est_disponivel >= qtd_saida:
                atendido = registrar_saida_fifo(p_id, qtd_saida, tipo_saida, motivo, usuario_ativo)
                
                # Registra no histórico diário da IA
                conn = get_db_connection()
                hoje_str = datetime.now().strftime("%Y-%m-%d")
                conn.execute('''
                    INSERT INTO historico_movimentacoes (produto_id, data, quantidade_saida, dia_semana, eh_fim_semana)
                    VALUES (?, ?, ?, ?, ?)
                ''', (p_id, hoje_str, qtd_saida, datetime.now().weekday(), 1 if datetime.now().weekday() >= 5 else 0))
                conn.commit()
                conn.close()
                
                st.success(f"Baixa de {atendido:.2f} {un} efetuada com sucesso seguindo a ordem de validade (FIFO)!")
                st.rerun()
            else:
                st.error("Quantidade solicitada excede o saldo em estoque.")

# -----------------------------------------------------------------------------
# 4. PREVISÃO DE IA & COMPRAS (RF09 AO RF13, RNF12)
# -----------------------------------------------------------------------------
elif menu == "🤖 Previsão de IA & Compras":
    st.title("🤖 Previsão de Demanda & Sugestão de Compras (IA)")
    st.caption("O modelo analisa padrões históricos de consumo e sazonalidades para calcular o volume ideal de compras.")
    
    conn = get_db_connection()
    produtos = conn.execute("SELECT produto_id, nome, unidade_medida, estoque_minimo FROM produtos").fetchall()
    
    dias_previsao = st.slider("🗓️ Selecione o horizonte futuro para análise (dias):", min_value=1, max_value=30, value=7)
    
    previsoes_lista = []
    graficos_previsao = {}
    
    for prod in produtos:
        p_id = prod['produto_id']
        nome_prod = prod['nome']
        un = prod['unidade_medida']
        est_min = prod['estoque_minimo']
        
        df_hist = pd.read_sql_query("SELECT quantidade_saida, dia_semana, eh_fim_semana FROM historico_movimentacoes WHERE produto_id = ?", conn, params=(p_id,))
        
        if len(df_hist) >= 10:
            X = df_hist[['dia_semana', 'eh_fim_semana']]
            y = df_hist['quantidade_saida']
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            hoje = datetime.now()
            datas_futuras = [hoje + timedelta(days=i) for i in range(1, dias_previsao + 1)]
            df_futuro = pd.DataFrame({
                'dia_semana': [d.weekday() for d in datas_futuras],
                'eh_fim_semana': [1 if d.weekday() >= 5 else 0 for d in datas_futuras]
            })
            
            demandas_diarias = model.predict(df_futuro)
            demanda_total = float(np.sum(demandas_diarias))
            
            estoque_atual = obter_estoque_atual(p_id)
            sugestao = max(0.0, (demanda_total + est_min) - estoque_atual)
            
            previsoes_lista.append({
                "ID": p_id,
                "Produto": nome_prod,
                "Estoque Atual": f"{estoque_atual:.2f} {un}",
                "Estoque Mínimo": f"{est_min:.2f} {un}",
                "Demanda Prevista": f"{demanda_total:.2f} {un}",
                "Sugestão de Compra": f"{sugestao:.2f} {un}" if sugestao > 0 else "Estoque OK",
                "Comprar_Num": sugestao
            })
            
            # Dados para o gráfico de tendência diária
            df_chart = pd.DataFrame({
                'Data': [d.strftime('%d/%m') for d in datas_futuras],
                'Demanda Prevista': demandas_diarias
            })
            graficos_previsao[nome_prod] = df_chart
            
    conn.close()
    
    if previsoes_lista:
        df_prev = pd.DataFrame(previsoes_lista)
        
        st.markdown("<div class='section-header'>🛒 Sugestões de Compras Geradas pela IA</div>", unsafe_allow_html=True)
        st.dataframe(
            df_prev[['Produto', 'Estoque Atual', 'Estoque Mínimo', 'Demanda Prevista', 'Sugestão de Compra']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("<div class='section-header'>📈 Tendência Diária de Demanda por Produto</div>", unsafe_allow_html=True)
        prod_chart_sel = st.selectbox("Selecione o produto para visualizar a projeção diária:", list(graficos_previsao.keys()))
        
        if prod_chart_sel in graficos_previsao:
            df_c = graficos_previsao[prod_chart_sel]
            fig_line = px.line(df_c, x='Data', y='Demanda Prevista', markers=True, title=f"Projeção Diária - {prod_chart_sel}")
            fig_line.update_traces(line_color='#28a745', line_width=3)
            fig_line.update_layout(template='plotly_white', height=300)
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.info("💡 **Como interpretar (RNF12):** O modelo calcula as vendas projetadas e subtrai o que já está guardado no estoque. Caso a 'Sugestão de Compra' seja maior que zero, faça a aquisição desse saldo para evitar desabastecimento.")

# -----------------------------------------------------------------------------
# 5. RELATÓRIOS & AUDITORIA (RF14)
# -----------------------------------------------------------------------------
elif menu == "📈 Relatórios & Auditoria":
    st.title("📈 Relatórios e Registros de Auditoria")
    st.caption("Acompanhe o histórico de movimentações do sistema e análise de perdas por desperdício.")
    
    conn = get_db_connection()
    
    r1, r2 = st.tabs(["📋 Histórico Completo de Movimentações", "🗑️ Análise de Desperdícios"])
    
    with r1:
        st.subheader("Histórico de Entradas, Saídas e Perdas")
        df_movs = pd.read_sql_query('''
            SELECT m.movimentacao_id as ID, p.nome as Produto, m.tipo_movimentacao as Tipo, 
                   m.quantidade as Qtd, p.unidade_medida as Unidade, 
                   COALESCE(m.motivo_desperdicio, '-') as Motivo, m.usuario as Responsavel, m.data_hora as Data
            FROM movimentacoes_estoque m
            JOIN lotes_estoque l ON m.lote_id = l.lote_id
            JOIN produtos p ON l.produto_id = p.produto_id
            ORDER BY m.data_hora DESC
        ''', conn)
        st.dataframe(df_movs, use_container_width=True, hide_index=True)
        
    with r2:
        st.subheader("Desperdício de Alimentos por Categoria e Motivo")
        df_desp = pd.read_sql_query('''
            SELECT p.nome as Produto, SUM(m.quantidade) as Total, m.motivo_desperdicio as Motivo
            FROM movimentacoes_estoque m
            JOIN lotes_estoque l ON m.lote_id = l.lote_id
            JOIN produtos p ON l.produto_id = p.produto_id
            WHERE m.tipo_movimentacao = 'DESPERDICIO'
            GROUP BY p.nome, m.motivo_desperdicio
        ''', conn)
        
        if not df_desp.empty:
            col_chart, col_data = st.columns([1.2, 1])
            with col_chart:
                fig_donut = px.pie(df_desp, names='Motivo', values='Total', hole=0.4, title="Distribuição dos Motivos de Perda")
                fig_donut.update_layout(template='plotly_white')
                st.plotly_chart(fig_donut, use_container_width=True)
            with col_data:
                st.dataframe(df_desp, use_container_width=True, hide_index=True)
        else:
            st.success("Excelente! Nenhum desperdício registrado até o momento.")
            
    conn.close()