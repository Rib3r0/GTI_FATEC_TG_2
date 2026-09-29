import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from database import get_db_connection, obter_estoque_atual

def gerar_previsao_demanda(dias_previsao=7):
    """
    Carrega o histórico de movimentações, treina o RandomForestRegressor para cada produto
    e calcula a sugestão exata de compra.
    """
    conn = get_db_connection()
    produtos = conn.execute("SELECT produto_id, nome, unidade_medida, estoque_minimo FROM produtos").fetchall()
    
    previsoes_lista = []
    graficos_previsao = {}
    
    for prod in produtos:
        p_id = prod['produto_id']
        nome_prod = prod['nome']
        un = prod['unidade_medida']
        est_min = prod['estoque_minimo']
        
        df_hist = pd.read_sql_query(
            "SELECT quantidade_saida, dia_semana, eh_fim_semana FROM historico_movimentacoes WHERE produto_id = ?", 
            conn, 
            params=(p_id,)
        )
        
        if len(df_hist) >= 10:
            X = df_hist[['dia_semana', 'eh_fim_semana']]
            y = df_hist['quantidade_saida']
            
            # Treinamento do Modelo de Regressão
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Cálculo dos dias futuros
            hoje = datetime.now()
            datas_futuras = [hoje + timedelta(days=i) for i in range(1, dias_previsao + 1)]
            df_futuro = pd.DataFrame({
                'dia_semana': [d.weekday() for d in datas_futuras],
                'eh_fim_semana': [1 if d.weekday() >= 5 else 0 for d in datas_futuras]
            })
            
            demandas_diarias = model.predict(df_futuro)
            demanda_total = float(np.sum(demandas_diarias))
            
            estoque_atual = obter_estoque_atual(p_id)
            
            # Fórmula de Sugestão de Compra: (Demanda Prevista + Estoque Mínimo) - Estoque Atual
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
            
            graficos_previsao[nome_prod] = pd.DataFrame({
                'Data': [d.strftime('%d/%m') for d in datas_futuras],
                'Demanda Prevista': demandas_diarias
            })
            
    conn.close()
    return pd.DataFrame(previsoes_lista), graficos_previsao