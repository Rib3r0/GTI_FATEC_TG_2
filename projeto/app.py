import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Importações dos Módulos Locais
import database as db
import ml_model as ml

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA E CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gestão de Estoque & IA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Inicializa o Banco de Dados se necessário
db.init_db()

# -----------------------------------------------------------------------------
# MENU LATERAL DE NAVEGAÇÃO
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
# 1. PAINEL DE CONTROLE / DASHBOARD
# -----------------------------------------------------------------------------
if menu == "📊 Painel de Controle (Dashboard)":
    st.title("📊 Painel Geral de Estoque e Alertas")
    st.caption("Acompanhamento de saldo, alertas automáticos e comparativos de estoque em tempo real.")
    
    conn = db.get_db_connection()
    df_prods = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    
    df_prods['estoque_atual'] = df_prods['produto_id'].apply(db.obter_estoque_atual)
    df_prods['status'] = np.where(df_prods['estoque_atual'] < df_prods['estoque_minimo'], "⚠️ Baixo", "✅ Ok")
    
    total_itens = len(df_prods)
    itens_criticos = len(df_prods[df_prods['estoque_atual'] < df_prods['estoque_minimo']])
    total_unidades = df_prods['estoque_atual'].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Produtos Cadastrados", total_itens)
    c2.metric("Itens Críticos (Abaixo do Mínimo)", itens_criticos, delta_color="inverse", delta=f"{itens_criticos} alerta(s)" if itens_criticos > 0 else "Estoque Seguro")
    c3.metric("Volume Total no Estoque", f"{total_unidades:.1f} Un/Kg")
    
    st.divider()
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("<div class='section-header'>📊 Comparativo: Estoque Atual vs Mínimo Exigido</div>", unsafe_allow_html=True)
        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(x=df_prods['nome'], y=df_prods['estoque_atual'], name='Estoque Atual', marker_color='#007bff'))
        fig_bars.add_trace(go.Bar(x=df_prods['nome'], y=df_prods['estoque_minimo'], name='Estoque Mínimo de Segurança', marker_color='#dc3545'))
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
            "estoque_atual": st.column_config.ProgressColumn("Nível de Estoque", format="%.1f", min_value=0, max_value=float(df_prods['estoque_atual'].max() * 1.2) if len(df_prods) > 0 else 100),
            "estoque_minimo": "Mínimo",
            "unidade_medida": "Unidade",
            "status": "Status"
        },
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------------------------------------------------
# 2. GESTÃO DE PRODUTOS
# -----------------------------------------------------------------------------
elif menu == "📦 Gestão de Produtos":
    st.title("📦 Cadastro e Gestão de Produtos")
    
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
                conn = db.get_db_connection()
                conn.execute("INSERT INTO produtos (nome, categoria, unidade_medida, estoque_minimo) VALUES (?, ?, ?, ?)", (nome, categoria, unidade, est_min))
                conn.commit()
                conn.close()
                st.success(f"Produto '{nome}' cadastrado com sucesso!")
                st.rerun()

    st.markdown("<div class='section-header'>Produtos Cadastrados no Sistema</div>", unsafe_allow_html=True)
    conn = db.get_db_connection()
    df_p = pd.read_sql_query("SELECT * FROM produtos", conn)
    conn.close()
    st.dataframe(df_p, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# 3. MOVIMENTAÇÃO & REGRA FIFO
# -----------------------------------------------------------------------------
elif menu == "🔄 Movimentação & Regra FIFO":
    st.title("🔄 Movimentação de Estoque")
    
    conn = db.get_db_connection()
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
            conn = db.get_db_connection()
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
        est_disponivel = db.obter_estoque_atual(p_id)
        
        st.info(f"💡 Saldo total disponível em lotes ativos: **{est_disponivel:.2f} {un}**")
        
        c1, c2 = st.columns(2)
        tipo_saida = c1.radio("Finalidade da Saída", ["SAIDA_CONSUMO", "DESPERDICIO"])
        qtd_saida = c2.number_input(f"Quantidade a dar baixa ({un})", min_value=0.1, max_value=float(est_disponivel) if est_disponivel > 0 else 1.0, value=1.0)
        
        motivo = None
        if tipo_saida == "DESPERDICIO":
            motivo = st.selectbox("Motivo da Perda (RF07)", ["Validade Vencida", "Avaria / Quebra", "Erro de Preparação", "Outros"])
            
        if st.button("📤 Confirmar Baixa de Estoque", type="primary"):
            if est_disponivel >= qtd_saida:
                atendido = db.registrar_saida_fifo(p_id, qtd_saida, tipo_saida, motivo, usuario_ativo)
                
                conn = db.get_db_connection()
                hoje_str = datetime.now().strftime("%Y-%m-%d")
                conn.execute('''
                    INSERT INTO historico_movimentacoes (produto_id, data, quantidade_saida, dia_semana, eh_fim_semana)
                    VALUES (?, ?, ?, ?, ?)
                ''', (p_id, hoje_str, qtd_saida, datetime.now().weekday(), 1 if datetime.now().weekday() >= 5 else 0))
                conn.commit()
                conn.close()
                
                st.success(f"Baixa de {atendido:.2f} {un} efetuada com sucesso pelo método FIFO!")
                st.rerun()
            else:
                st.error("Quantidade solicitada excede o saldo em estoque.")

# -----------------------------------------------------------------------------
# 4. PREVISÃO DE IA & COMPRAS
# -----------------------------------------------------------------------------
elif menu == "🤖 Previsão de IA & Compras":
    st.title("🤖 Previsão de Demanda & Sugestão de Compras (IA)")
    
    dias_previsao = st.slider("🗓️ Selecione o horizonte futuro para análise (dias):", min_value=1, max_value=30, value=7)
    
    df_prev, graficos_previsao = ml.gerar_previsao_demanda(dias_previsao)
    
    if not df_prev.empty:
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
            
        st.info("💡 **Como interpretar (RNF12):** O modelo calcula as vendas projetadas e subtrai o saldo atual. Caso a 'Sugestão de Compra' seja maior que zero, efetue a aquisição do saldo recomendado.")

# -----------------------------------------------------------------------------
# 5. RELATÓRIOS & AUDITORIA
# -----------------------------------------------------------------------------
elif menu == "📈 Relatórios & Auditoria":
    st.title("📈 Relatórios e Registros de Auditoria")
    
    conn = db.get_db_connection()
    r1, r2 = st.tabs(["📋 Histórico Completo de Movimentações", "🗑️ Análise de Desperdícios"])
    
    with r1:
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
                st.plotly_chart(fig_donut, use_container_width=True)
            with col_data:
                st.dataframe(df_desp, use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum desperdício registrado no momento.")
            
    conn.close()