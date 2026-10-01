import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.config import RAW_DIR, DIAS_TREINO, SEED

np.random.seed(SEED)

PRATOS = {
    "feijoada":     {"base": 40, "fds_mult": 1.8, "tendencia": 0.02},
    "strogonoff":   {"base": 35, "fds_mult": 1.5, "tendencia": 0.01},
    "frango_grelhado": {"base": 45, "fds_mult": 1.2, "tendencia": 0.03},
    "lasanha":      {"base": 30, "fds_mult": 1.6, "tendencia": 0.00},
    "salada_caesar": {"base": 20, "fds_mult": 1.1, "tendencia": 0.04},
}

INSUMOS = {
    "arroz":        {"unidade": "kg", "custo": 5.0},
    "feijao":       {"unidade": "kg", "custo": 8.0},
    "carne_bovina": {"unidade": "kg", "custo": 35.0},
    "frango":       {"unidade": "kg", "custo": 15.0},
    "batata":       {"unidade": "kg", "custo": 6.0},
    "tomate":       {"unidade": "kg", "custo": 7.0},
    "cebola":       {"unidade": "kg", "custo": 4.0},
    "alface":       {"unidade": "kg", "custo": 5.0},
    "queijo":       {"unidade": "kg", "custo": 40.0},
    "farinha":      {"unidade": "kg", "custo": 4.0},
}

FICHA_TECNICA = {
    "feijoada":        {"feijao": 0.25, "carne_bovina": 0.30, "arroz": 0.15, "cebola": 0.05},
    "strogonoff":      {"frango": 0.25, "arroz": 0.15, "cebola": 0.05},
    "frango_grelhado": {"frango": 0.30, "arroz": 0.15, "salada": 0.0, "alface": 0.05, "tomate": 0.05, "batata": 0.15},
    "lasanha":         {"carne_bovina": 0.25, "queijo": 0.15, "farinha": 0.05, "tomate": 0.10},
    "salada_caesar":   {"alface": 0.15, "frango": 0.10, "queijo": 0.05, "tomate": 0.05},
}

def calcular_pascoa(ano: int) -> date:
    """Algoritmo de Meeus/Jones/Butcher — Páscoa para qualquer ano gregoriano."""
    a = ano % 19
    b = ano // 100
    c = ano % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


def feriados_nacionais(ano: int) -> set[date]:
    pascoa = calcular_pascoa(ano)
    return {
        date(ano, 1, 1),                    # Confraternização Universal
        pascoa - timedelta(days=48),        # Carnaval (segunda)
        pascoa - timedelta(days=47),        # Carnaval (terça)
        pascoa - timedelta(days=2),         # Sexta-feira Santa
        date(ano, 4, 21),                   # Tiradentes
        date(ano, 5, 1),                    # Dia do Trabalho
        pascoa + timedelta(days=60),        # Corpus Christi
        date(ano, 9, 7),                    # Independência
        date(ano, 10, 12),                  # N. Sra. Aparecida
        date(ano, 11, 2),                   # Finados
        date(ano, 11, 15),                  # Proclamação da República
        date(ano, 11, 20),                  # Consciência Negra
        date(ano, 12, 25),                  # Natal
    }


def feriados_no_intervalo(inicio: date, fim: date) -> set[date]:
    feriados: set[date] = set()
    for ano in range(inicio.year, fim.year + 1):
        feriados |= feriados_nacionais(ano)
    return {f for f in feriados if inicio <= f <= fim}

def gerar_vendas(inicio: date = None, dias: int = DIAS_TREINO) -> pd.DataFrame:
    if inicio is None:
        inicio = date.today() - timedelta(days=dias)

    fim = inicio + timedelta(days=dias - 1)
    feriados = feriados_no_intervalo(inicio, fim)
    
    linhas = []

    for d in range(dias):
        dia = inicio + timedelta(days=d)
        dow = dia.weekday()  # 0=seg ... 6=dom
        is_fds = dow >= 5
        is_feriado = dia in feriados

        for prato, cfg in PRATOS.items():
            base = cfg["base"]
            mult_fds = cfg["fds_mult"] if is_fds else 1.0
            mult_feriado = 1.4 if is_feriado else 1.0
            tendencia = 1 + cfg["tendencia"] * (d / 30)
            sazonal = 1 + 0.1 * np.sin(2 * np.pi * d / 90)  # leve ciclo trimestral
            ruido = np.random.normal(1.0, 0.15)

            qtd = base * mult_fds * mult_feriado * tendencia * sazonal * ruido
            qtd = max(0, int(round(qtd)))

            linhas.append({
                "data": dia,
                "prato": prato,
                "quantidade": qtd,
                "feriado": int(is_feriado),
                "fim_de_semana": int(is_fds),
            })

    return pd.DataFrame(linhas)


def gerar_cadastros() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pratos_df = pd.DataFrame(
        [{"id": i, "nome": p} for i, p in enumerate(PRATOS.keys(), start=1)]
    )
    insumos_df = pd.DataFrame(
        [{"id": i, "nome": k, "unidade": v["unidade"], "custo_unitario": v["custo"]}
         for i, (k, v) in enumerate(INSUMOS.items(), start=1)]
    )
    ficha = []
    for prato, insumos in FICHA_TECNICA.items():
        for insumo, qtd in insumos.items():
            if qtd > 0:
                ficha.append({"prato": prato, "insumo": insumo, "quantidade_por_prato": qtd})
    ficha_df = pd.DataFrame(ficha)
    return pratos_df, insumos_df, ficha_df


def salvar_dados():
    vendas = gerar_vendas()
    pratos, insumos, ficha = gerar_cadastros()

    vendas.to_csv(RAW_DIR / "vendas.csv", index=False)
    pratos.to_csv(RAW_DIR / "pratos.csv", index=False)
    insumos.to_csv(RAW_DIR / "insumos.csv", index=False)
    ficha.to_csv(RAW_DIR / "ficha_tecnica.csv", index=False)

    print(f"[OK] vendas: {len(vendas)} linhas")
    print(f"[OK] pratos: {len(pratos)} linhas")
    print(f"[OK] insumos: {len(insumos)} linhas")
    print(f"[OK] ficha técnica: {len(ficha)} linhas")


if __name__ == "__main__":
    salvar_dados()