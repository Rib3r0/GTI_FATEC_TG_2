import sqlite3

DB_NAME = "gestao_estoque.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Cria as tabelas e popula dados iniciais de teste caso esteja vazio."""
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
    
    # RF03, RF04 - Lotes (FIFO)
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
    
    # RF09, RF10, RF13 - Histórico para a IA
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
    
    # Injeção inicial se o banco estiver limpo
    cursor.execute("SELECT COUNT(*) FROM produtos")
    if cursor.fetchone()[0] == 0:
        import numpy as np
        from datetime import datetime, timedelta
        
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

def obter_estoque_atual(produto_id):
    """Retorna a soma da quantidade disponível em todos os lotes ativos do produto."""
    conn = get_db_connection()
    res = conn.execute("SELECT SUM(quantidade_atual) FROM lotes_estoque WHERE produto_id = ?", (produto_id,)).fetchone()[0]
    conn.close()
    return res if res else 0.0

def registrar_saida_fifo(produto_id, qtd_solicitada, tipo_mov, motivo, usuario):
    """Aplica a baixa de estoque consorciada ao modelo FIFO (Primeiro que Vence, Primeiro que Sai)."""
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