import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.config import RAW_DIR, DIAS_TREINO, SEED

np.random.seed(SEED)

# ------------------------------------------------------------------
# Perfis de cada prato
# ------------------------------------------------------------------
PRATOS = {
    "feijoada": {
        "base": 40,
        "dow": [0.85, 0.90, 0.95, 1.00, 1.15, 1.85, 1.75],
        # forte no inverno, fraco no verão
        "mes": [0.55, 0.60, 0.80, 1.00, 1.30, 1.65, 1.75, 1.55, 1.15, 0.85, 0.65, 0.55],
        "tendencia_mensal": -0.005,
        "weather_sens": -0.030,
        "promo_sens": 1.45,
        "random_walk_std": 0.010,
    },
    "strogonoff": {
        "base": 35,
        "dow": [1.00, 1.00, 1.05, 1.05, 1.20, 1.35, 1.30],
        "mes": [1.15, 1.15, 1.05, 1.00, 0.90, 0.85, 0.85, 0.90, 1.05, 1.15, 1.20, 1.20],
        "tendencia_mensal": -0.012,   # em declínio
        "weather_sens": -0.005,
        "promo_sens": 1.30,
        "random_walk_std": 0.012,
    },
    "frango_grelhado": {
        "base": 45,
        "dow": [1.05, 1.05, 1.05, 1.05, 1.10, 1.15, 1.10],
        "mes": [1.00, 1.00, 1.00, 1.05, 1.05, 1.05, 1.00, 0.95, 1.00, 1.05, 1.10, 1.10],
        "tendencia_mensal": 0.028,    # crescimento forte
        "weather_sens": 0.010,
        "promo_sens": 1.25,
        "random_walk_std": 0.006,
    },
    "lasanha": {
        "base": 30,
        "dow": [0.95, 1.00, 1.00, 1.05, 1.20, 1.55, 1.50],
        "mes": [1.00, 1.00, 1.05, 1.10, 1.15, 1.25, 1.20, 1.10, 1.00, 0.95, 0.95, 1.00],
        "tendencia_mensal": -0.008,
        "weather_sens": -0.015,
        "promo_sens": 1.40,
        "random_walk_std": 0.010,
    },
    "salada_caesar": {
        "base": 20,
        "dow": [1.15, 1.10, 1.15, 1.10, 1.10, 0.85, 0.75],
        # forte no verão, quase some no inverno
        "mes": [1.80, 1.70, 1.40, 1.05, 0.75, 0.45, 0.35, 0.50, 0.85, 1.25, 1.55, 1.75],
        "tendencia_mensal": 0.020,
        "weather_sens": 0.040,
        "promo_sens": 1.20,
        "random_walk_std": 0.015,
    },
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
    "strogonoff":      {"frango": 0.25, "arroz": 0.15, "cebola": 0.05, "batata": 0.10},
    "frango_grelhado": {"frango": 0.30, "arroz": 0.15, "alface": 0.05, "tomate": 0.05, "batata": 0.15},
    "lasanha":         {"carne_bovina": 0.25, "queijo": 0.15, "farinha": 0.05, "tomate": 0.10},
    "salada_caesar":   {"alface": 0.15, "frango": 0.10, "queijo": 0.05, "tomate": 0.05},
}


# ------------------------------------------------------------------
# Feriados (mesma lógica já implementada)
# ------------------------------------------------------------------
def calcular_pascoa(ano: int) -> date:
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
        date(ano, 1, 1),
        pascoa - timedelta(days=48),
        pascoa - timedelta(days=47),
        pascoa - timedelta(days=2),
        date(ano, 4, 21),
        date(ano, 5, 1),
        pascoa + timedelta(days=60),
        date(ano, 9, 7),
        date(ano, 10, 12),
        date(ano, 11, 2),
        date(ano, 11, 15),
        date(ano, 11, 20),
        date(ano, 12, 25),
    }


def feriados_no_intervalo(inicio: date, fim: date) -> set[date]:
    feriados: set[date] = set()
    for ano in range(inicio.year, fim.year + 1):
        feriados |= feriados_nacionais(ano)
    return {f for f in feriados if inicio <= f <= fim}


# ------------------------------------------------------------------
# Simulação ambiental
# ------------------------------------------------------------------
def _temperatura_serie(datas: list[date], seed: int) -> np.ndarray:
    """Temperatura sintética: ciclo anual + ruído + persistência."""
    rng = np.random.default_rng(seed)
    temps = []
    for d in datas:
        doy = d.timetuple().tm_yday
        # mais frio em jun-jul, mais quente em jan
        base = 22 + 6 * np.sin(2 * np.pi * (doy - 15) / 365)
        temps.append(base)
    temps = np.array(temps)
    # ruído autocorrelacionado
    ruido = np.zeros(len(temps))
    for t in range(1, len(temps)):
        ruido[t] = 0.6 * ruido[t - 1] + rng.normal(0, 1.8)
    return np.round(temps + ruido, 1)


def _promocoes(datas: list[date], pratos: list[str], seed: int) -> dict:
    """Retorna {(data, prato): bool} para dias de promoção."""
    rng = np.random.default_rng(seed + 1)
    promos = {}
    ativa = None
    restam = 0
    for d in datas:
        if restam > 0:
            promos[(d, ativa)] = True
            restam -= 1
        else:
            if rng.random() < 0.02:  # 2% de chance por dia
                ativa = rng.choice(pratos)
                restam = int(rng.integers(2, 5))
                promos[(d, ativa)] = True
                restam -= 1
    return promos


# ------------------------------------------------------------------
# Geração principal
# ------------------------------------------------------------------
def gerar_vendas(inicio: date = None, dias: int = DIAS_TREINO) -> pd.DataFrame:
    if inicio is None:
        inicio = date.today() - timedelta(days=dias)
    fim = inicio + timedelta(days=dias - 1)

    datas = [inicio + timedelta(days=i) for i in range(dias)]
    feriados = feriados_no_intervalo(inicio, fim)
    temperaturas = _temperatura_serie(datas, SEED)
    promos = _promocoes(datas, list(PRATOS.keys()), SEED)

    dia_regime = int(dias * 0.6)

    rng = np.random.default_rng(SEED + 2)

    # Ruído AR(1) por prato — persistente
    ruidos = {p: np.zeros(dias) for p in PRATOS}
    for p in PRATOS:
        for t in range(1, dias):
            ruidos[p][t] = 0.45 * ruidos[p][t - 1] + rng.normal(0, 0.12)

    # Random walk por prato — drift lento que muda a composição
    rws = {p: np.zeros(dias) for p in PRATOS}
    for p, cfg in PRATOS.items():
        std = cfg.get("random_walk_std", 0.008)
        for t in range(1, dias):
            rws[p][t] = rws[p][t - 1] + rng.normal(0, std)

    linhas = []
    for i, dia in enumerate(datas):
        dow = dia.weekday()
        mes_idx = dia.month - 1
        temp = temperaturas[i]
        is_fds = int(dow >= 5)
        is_feriado = dia in feriados

        for prato, cfg in PRATOS.items():
            mult_dow = cfg["dow"][dow]
            mult_mes = cfg["mes"][mes_idx]
            mult_tend = 1 + cfg["tendencia_mensal"] * (i / 30)

            desvio_temp = temp - 22
            mult_clima = 1 + cfg["weather_sens"] * desvio_temp

            mult_feriado = 1.25 if is_feriado else 1.0

            em_promo = (dia, prato) in promos
            mult_promo = cfg["promo_sens"] if em_promo else 1.0

            mult_regime = 1.0
            if i >= dia_regime:
                if prato == "strogonoff":
                    mult_regime = 0.85    # cai após regime
                elif prato == "salada_caesar":
                    mult_regime = 1.15    # sobe após regime

            mult_ruido = float(np.exp(ruidos[prato][i]))
            mult_rw = float(np.exp(rws[prato][i]))

            qtd = (
                cfg["base"]
                * mult_dow
                * mult_mes
                * mult_tend
                * mult_clima
                * mult_feriado
                * mult_promo
                * mult_regime
                * mult_ruido
                * mult_rw
            )

            if rng.random() < 0.015:
                qtd *= rng.uniform(1.5, 2.3)

            qtd = max(0, int(round(qtd)))

            linhas.append({
                "data": dia,
                "prato": prato,
                "quantidade": qtd,
                "feriado": int(is_feriado),
                "fim_de_semana": is_fds,
                "temperatura": temp,
                "promocao": int(em_promo),
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
    return pratos_df, insumos_df, pd.DataFrame(ficha)


def salvar_dados():
    vendas = gerar_vendas()
    pratos, insumos, ficha = gerar_cadastros()

    vendas.to_csv(RAW_DIR / "vendas.csv", index=False)
    pratos.to_csv(RAW_DIR / "pratos.csv", index=False)
    insumos.to_csv(RAW_DIR / "insumos.csv", index=False)
    ficha.to_csv(RAW_DIR / "ficha_tecnica.csv", index=False)

    print(f"[OK] vendas: {len(vendas)} linhas | período: {vendas['data'].min()} a {vendas['data'].max()}")
    print(f"[OK] pratos: {len(pratos)} linhas")
    print(f"[OK] insumos: {len(insumos)} linhas")
    print(f"[OK] ficha técnica: {len(ficha)} linhas")


if __name__ == "__main__":
    salvar_dados()