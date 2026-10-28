"""Gerador de dados sintéticos de vendas de um restaurante (versão realista).

Mantém a mesma interface da versão anterior (gerar_vendas, gerar_cadastros,
salvar_dados, feriados_no_intervalo) e as mesmas colunas de saída.

CRITÉRIOS DE REALISMO USADOS
  Clima
    1. Temperatura com sazonalidade correta (quente em jan, fria em jul).
    2. Frentes frias (mais no inverno) e ondas de calor (mais no verão).
    3. Chuva simulada (mais no verão, persistente, mais provável em frente fria).
    4. O efeito do clima usa a ANOMALIA térmica (temp - média do dia do ano),
       para não contar a sazonalidade duas vezes.
  Calendário
    5. Dia da semana por prato; em feriado de lazer, perfil de domingo.
    6. Sazonalidade mensal suavizada (sem degraus) e com amplitude moderada.
    7. Feriados nacionais + 9 de Julho (SP), com efeito por tipo.
    8. Natal/Ano Novo (casa quase vazia), Carnaval e Sexta-feira Santa (queda).
    9. Dias de ponte (sexta após feriado de quinta, segunda antes de terça).
   10. Véspera de Natal e de Ano Novo com movimento menor.
   11. Datas comemorativas: Dia das Mães, dos Pais, dos Namorados e Páscoa.
   12. Confraternizações de fim de ano (1 a 23/dez, dias úteis).
   13. Efeito do dia de pagamento (dias 5-8 e 20-23 sobem; fim/início do mês caem).
  Demanda
   14. Tendência composta (exponencial) por prato + crescimento geral lento.
   15. Mudança de cardápio gradual (transição de ~3 semanas, não degrau).
   16. Popularidade que oscila devagar (processo mean-reverting, não passeio aleatório).
   17. Movimento comum a todos os pratos (clientes do dia) + ruído próprio de cada prato.
   18. Contagem com sobredispersão (Poisson-Gama / binomial negativa).
   19. Promoções mais frequentes em dias fracos, com intensidade variável,
       canibalização dos outros pratos e queda leve depois da promoção.
   20. Eventos pontuais (grupos, aniversários) que elevam um prato no dia.
   21. Rupturas (acabou o prato): venda menor que a demanda.
  Cadastros
   22. Porções por prato em quantidade crua realista.
   23. Custos de insumos em valores de referência atuais (aproximados).
"""
import numpy as np
import pandas as pd
from calendar import monthrange
from datetime import date, timedelta
from src.config import RAW_DIR, DIAS_TREINO, SEED

np.random.seed(SEED)  # compatibilidade; a geração usa Generators próprios

# ------------------------------------------------------------------
# Parâmetros globais
# ------------------------------------------------------------------
FORCA_SAZONALIDADE = 0.75        # 1.0 = tabela mensal original; menor = mais suave
CRESCIMENTO_GERAL_MENSAL = 0.004  # crescimento do restaurante (~0,4% ao mês)
RUIDO_COMUM_STD = 0.04           # "movimento do dia" compartilhado por todos os pratos
RUIDO_PRATO_STD = 0.06           # ruído próprio de cada prato
RUIDO_AR = 0.50                  # persistência dos ruídos de um dia para o outro
POPULARIDADE_STD = 0.025         # oscilação lenta da popularidade (desvio estacionário)
DISPERSAO_NB = 100.0             # maior = mais próximo de Poisson
P_EVENTO = 0.012                 # chance diária de evento pontual por prato
P_RUPTURA = 0.010                # chance diária de acabar o prato
DIAS_MUDANCA_CARDAPIO = 21       # duração da transição de cardápio
FATOR_CHUVA_MOVIMENTO = 0.93     # chuva reduz o movimento geral
P_PROMO_DIA_SEMANA = [0.035, 0.035, 0.035, 0.025, 0.012, 0.010, 0.010]  # seg..dom

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
        "feriado_lazer": 1.35,
        "chuva": 1.06,
    },
    "strogonoff": {
        "base": 35,
        "dow": [1.00, 1.00, 1.05, 1.05, 1.20, 1.35, 1.30],
        "mes": [1.15, 1.15, 1.05, 1.00, 0.90, 0.85, 0.85, 0.90, 1.05, 1.15, 1.20, 1.20],
        "tendencia_mensal": -0.012,   # em declínio
        "weather_sens": -0.005,
        "promo_sens": 1.30,
        "random_walk_std": 0.012,
        "feriado_lazer": 1.20,
        "chuva": 1.03,
    },
    "frango_grelhado": {
        "base": 45,
        "dow": [1.05, 1.05, 1.05, 1.05, 1.10, 1.15, 1.10],
        "mes": [1.00, 1.00, 1.00, 1.05, 1.05, 1.05, 1.00, 0.95, 1.00, 1.05, 1.10, 1.10],
        "tendencia_mensal": 0.018,    # crescimento (prato em alta)
        "weather_sens": 0.010,
        "promo_sens": 1.25,
        "random_walk_std": 0.006,
        "feriado_lazer": 1.05,
        "chuva": 1.00,
    },
    "lasanha": {
        "base": 30,
        "dow": [0.95, 1.00, 1.00, 1.05, 1.20, 1.55, 1.50],
        "mes": [1.00, 1.00, 1.05, 1.10, 1.15, 1.25, 1.20, 1.10, 1.00, 0.95, 0.95, 1.00],
        "tendencia_mensal": -0.008,
        "weather_sens": -0.015,
        "promo_sens": 1.40,
        "random_walk_std": 0.010,
        "feriado_lazer": 1.35,
        "chuva": 1.05,
    },
    "salada_caesar": {
        "base": 20,
        "dow": [1.15, 1.10, 1.15, 1.10, 1.10, 0.85, 0.75],
        # forte no verão, quase some no inverno
        "mes": [1.80, 1.70, 1.40, 1.05, 0.75, 0.45, 0.35, 0.50, 0.85, 1.25, 1.55, 1.75],
        "tendencia_mensal": 0.005,
        "weather_sens": 0.040,
        "promo_sens": 1.20,
        "random_walk_std": 0.015,
        "feriado_lazer": 0.90,
        "chuva": 0.90,
    },
}

# Mudança de cardápio no meio do período (fator final, aplicado de forma gradual)
REGIME_FINAL = {"strogonoff": 0.85, "salada_caesar": 1.15}

# Efeito das datas comemorativas por prato
ESPECIAIS = {
    "dia_das_maes":      {"feijoada": 1.35, "strogonoff": 1.35, "frango_grelhado": 1.20, "lasanha": 1.60, "salada_caesar": 1.10},
    "dia_dos_pais":      {"feijoada": 1.50, "strogonoff": 1.15, "frango_grelhado": 1.05, "lasanha": 1.25, "salada_caesar": 0.95},
    "dia_dos_namorados": {"feijoada": 0.90, "strogonoff": 1.30, "frango_grelhado": 1.10, "lasanha": 1.45, "salada_caesar": 1.20},
    "pascoa":            {"feijoada": 0.85, "strogonoff": 1.00, "frango_grelhado": 1.00, "lasanha": 1.30, "salada_caesar": 1.00},
}
PRATOS_CONFRATERNIZACAO = {"strogonoff", "lasanha", "frango_grelhado"}

# Probabilidade de chuva por mês (base)
CHUVA_MES = [0.50, 0.50, 0.40, 0.28, 0.20, 0.14, 0.12, 0.14, 0.25, 0.35, 0.42, 0.50]

# Valores de referência aproximados (R$/kg). Ajuste conforme seu fornecedor.
INSUMOS = {
    "arroz":        {"unidade": "kg", "custo": 5.5},
    "feijao":       {"unidade": "kg", "custo": 9.0},
    "carne_bovina": {"unidade": "kg", "custo": 38.0},
    "frango":       {"unidade": "kg", "custo": 17.0},
    "batata":       {"unidade": "kg", "custo": 6.5},
    "tomate":       {"unidade": "kg", "custo": 8.0},
    "cebola":       {"unidade": "kg", "custo": 5.0},
    "alface":       {"unidade": "kg", "custo": 9.0},
    "queijo":       {"unidade": "kg", "custo": 42.0},
    "farinha":      {"unidade": "kg", "custo": 4.5},
}

# Quantidade CRUA por prato (kg)
FICHA_TECNICA = {
    "feijoada":        {"feijao": 0.10, "carne_bovina": 0.20, "arroz": 0.08, "cebola": 0.03},
    "strogonoff":      {"frango": 0.18, "arroz": 0.08, "cebola": 0.03, "batata": 0.10},
    "frango_grelhado": {"frango": 0.20, "arroz": 0.08, "alface": 0.04, "tomate": 0.04, "batata": 0.15},
    "lasanha":         {"carne_bovina": 0.12, "queijo": 0.10, "farinha": 0.08, "tomate": 0.10},
    "salada_caesar":   {"alface": 0.15, "frango": 0.10, "queijo": 0.03, "tomate": 0.04},
}


# ------------------------------------------------------------------
# Feriados
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


def _feriados_nomeados(ano: int) -> dict[date, str]:
    """Feriados do ano com o TIPO de efeito sobre as vendas."""
    pascoa = calcular_pascoa(ano)
    return {
        date(ano, 1, 1): "fechado_parcial",
        pascoa - timedelta(days=48): "carnaval",
        pascoa - timedelta(days=47): "carnaval",
        pascoa - timedelta(days=2): "sexta_santa",
        date(ano, 4, 21): "lazer",
        date(ano, 5, 1): "lazer",
        pascoa + timedelta(days=60): "lazer",      # Corpus Christi
        date(ano, 7, 9): "lazer",                  # Revolução Constitucionalista (SP)
        date(ano, 9, 7): "lazer",
        date(ano, 10, 12): "lazer",
        date(ano, 11, 2): "lazer",
        date(ano, 11, 15): "lazer",
        date(ano, 11, 20): "lazer",
        date(ano, 12, 25): "fechado_parcial",
    }


def feriados_no_intervalo(inicio: date, fim: date) -> set[date]:
    """Feriados nacionais + 9 de Julho (SP). Usado no treino e na previsão."""
    feriados: set[date] = set()
    for ano in range(inicio.year, fim.year + 1):
        feriados |= set(_feriados_nomeados(ano))
    return {f for f in feriados if inicio <= f <= fim}


def _enesimo_dia_semana(ano: int, mes: int, dow: int, n: int) -> date:
    """n-ésimo dia da semana `dow` (0=seg) do mês. Ex.: 2º domingo de maio."""
    d = date(ano, mes, 1)
    d += timedelta(days=(dow - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _datas_especiais(ano: int) -> dict[date, str]:
    return {
        _enesimo_dia_semana(ano, 5, 6, 2): "dia_das_maes",
        _enesimo_dia_semana(ano, 8, 6, 2): "dia_dos_pais",
        date(ano, 6, 12): "dia_dos_namorados",
        calcular_pascoa(ano): "pascoa",
    }


def _dias_de_ponte(nomeados: dict[date, str]) -> set[date]:
    """Sexta após feriado de quinta e segunda antes de feriado de terça."""
    pontes = set()
    for d in nomeados:
        if d.weekday() == 3:
            pontes.add(d + timedelta(days=1))
        elif d.weekday() == 1:
            pontes.add(d - timedelta(days=1))
    return {p for p in pontes if p not in nomeados}


# ------------------------------------------------------------------
# Simulação ambiental
# ------------------------------------------------------------------
def _ar1(n: int, rho: float, sd: float, rng) -> np.ndarray:
    """Processo AR(1) com desvio-padrão estacionário `sd`."""
    x = np.zeros(n)
    x[0] = rng.normal(0, sd)
    inov = sd * np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0, inov)
    return x


def _temp_climatologica(doy: int) -> float:
    """Média diária esperada (estilo Sudeste/SP): pico em fim de janeiro, mínimo em julho."""
    return 20.8 + 3.6 * np.cos(2 * np.pi * (doy - 25) / 365)


def _temperatura_serie(datas: list[date], seed: int) -> np.ndarray:
    """Temperatura: ciclo anual correto + frentes frias + ondas de calor + ruído."""
    rng = np.random.default_rng(seed)
    n = len(datas)
    clima = np.array([_temp_climatologica(d.timetuple().tm_yday) for d in datas])
    ruido = _ar1(n, 0.6, 1.5, rng)

    episodio = np.zeros(n)
    t = 0
    while t < n:
        inverno = datas[t].month in (4, 5, 6, 7, 8, 9)
        p_frente = 0.035 if inverno else 0.010
        p_calor = 0.005 if inverno else 0.020
        u = rng.random()
        if u < p_frente:
            dur, delta = int(rng.integers(2, 6)), -rng.uniform(3, 8)
        elif u < p_frente + p_calor:
            dur, delta = int(rng.integers(3, 8)), rng.uniform(3, 7)
        else:
            t += 1
            continue
        episodio[t:t + dur] += delta
        t += dur
    episodio = np.convolve(episodio, [0.25, 0.5, 0.25], mode="same")  # bordas suaves

    return np.round(np.clip(clima + ruido + episodio, 4, 37), 1)


def _chuva_serie(datas: list[date], temps: np.ndarray, seed: int) -> np.ndarray:
    """Dia chuvoso (0/1): sazonal, persistente e mais provável em frente fria."""
    rng = np.random.default_rng(seed + 3)
    chuva = np.zeros(len(datas), dtype=int)
    for i, d in enumerate(datas):
        anomalia = temps[i] - _temp_climatologica(d.timetuple().tm_yday)
        p = CHUVA_MES[d.month - 1] * 0.75
        if i > 0 and chuva[i - 1]:
            p += 0.30
        if anomalia <= -3:
            p += 0.20
        chuva[i] = int(rng.random() < min(p, 0.9))
    return chuva


def _promocoes(datas: list[date], pratos: list[str], seed: int) -> dict:
    """Retorna {(data, prato): True} para dias de promoção (mais comuns em dias fracos)."""
    rng = np.random.default_rng(seed + 1)
    promos = {}
    ativa = None
    restam = 0
    for d in datas:
        if restam > 0:
            promos[(d, ativa)] = True
            restam -= 1
        elif rng.random() < P_PROMO_DIA_SEMANA[d.weekday()]:
            ativa = str(rng.choice(pratos))
            restam = int(rng.integers(2, 5))
            promos[(d, ativa)] = True
            restam -= 1
    return promos


# ------------------------------------------------------------------
# Fatores de demanda
# ------------------------------------------------------------------
def _fator_mes(tabela: list[float], dia: date) -> float:
    """Sazonalidade mensal interpolada entre os centros dos meses (sem degraus)."""
    n = monthrange(dia.year, dia.month)[1]
    pos = (dia.month - 1) + (dia.day - 0.5) / n - 0.5   # 0 = centro de janeiro
    base = np.floor(pos)
    frac = pos - base
    i0, i1 = int(base) % 12, (int(base) + 1) % 12
    v = tabela[i0] * (1 - frac) + tabela[i1] * frac
    return 1 + FORCA_SAZONALIDADE * (v - 1)


def _fator_pagamento(dia: date) -> float:
    d = dia.day
    if 5 <= d <= 8:
        return 1.05
    if 20 <= d <= 23:
        return 1.04
    if d >= 28 or d <= 2:
        return 0.96
    return 1.0


# ------------------------------------------------------------------
# Geração principal
# ------------------------------------------------------------------
def gerar_vendas(inicio: date = None, dias: int = DIAS_TREINO) -> pd.DataFrame:
    if inicio is None:
        inicio = date.today() - timedelta(days=dias)   # termina ontem
    fim = inicio + timedelta(days=dias - 1)

    datas = [inicio + timedelta(days=i) for i in range(dias)]
    pratos = list(PRATOS)

    feriados = feriados_no_intervalo(inicio, fim)
    nomeados: dict[date, str] = {}
    especiais: dict[date, str] = {}
    for ano in range(inicio.year - 1, fim.year + 2):
        nomeados.update(_feriados_nomeados(ano))
        especiais.update(_datas_especiais(ano))
    pontes = _dias_de_ponte(nomeados)

    temperaturas = _temperatura_serie(datas, SEED)
    chuva = _chuva_serie(datas, temperaturas, SEED)
    promos = _promocoes(datas, pratos, SEED)

    rng = np.random.default_rng(SEED + 2)

    # Processos latentes
    comum = _ar1(dias, RUIDO_AR, RUIDO_COMUM_STD, rng)                      # clientes do dia
    idio = {p: _ar1(dias, RUIDO_AR, RUIDO_PRATO_STD, rng) for p in pratos}  # ruído do prato
    popular = {p: _ar1(dias, 0.99, POPULARIDADE_STD, rng) for p in pratos}  # popularidade lenta

    # Mudança de cardápio gradual (logística) no ponto 60% do período
    dia_regime = int(dias * 0.6)
    largura = max(DIAS_MUDANCA_CARDAPIO / 4, 1)

    linhas = []
    for i, dia in enumerate(datas):
        dow = dia.weekday()
        fds = int(dow >= 5)
        eh_feriado = dia in feriados
        tipo_fer = nomeados.get(dia)
        especial = especiais.get(dia)
        vespera_fechada = nomeados.get(dia + timedelta(days=1)) == "fechado_parcial"
        temp = float(temperaturas[i])
        anomalia = float(np.clip(temp - _temp_climatologica(dia.timetuple().tm_yday), -10, 10))
        chove = bool(chuva[i])

        # fatores comuns a todos os pratos neste dia
        f_comum = (
            _fator_pagamento(dia)
            * (0.92 if dia in pontes else 1.0)
            * (0.80 if vespera_fechada else 1.0)
            * (FATOR_CHUVA_MOVIMENTO if chove else 1.0)
            * float(np.exp(comum[i]))
        )
        promo_hoje = {p for p in pratos if (dia, p) in promos}
        w_regime = 1 / (1 + np.exp(-(i - dia_regime) / largura))

        for prato in pratos:
            cfg = PRATOS[prato]

            # dia da semana (feriado de lazer em dia útil se comporta como domingo)
            mult_dow = cfg["dow"][dow]
            if tipo_fer == "lazer" and dow < 5:
                mult_dow = 0.5 * mult_dow + 0.5 * cfg["dow"][6]

            mult_mes = _fator_mes(cfg["mes"], dia)
            mult_tend = float(np.exp((cfg["tendencia_mensal"] + CRESCIMENTO_GERAL_MENSAL) * (i / 30)))
            mult_clima = float(np.exp(cfg["weather_sens"] * anomalia))
            mult_chuva = cfg["chuva"] if chove else 1.0

            # feriados por tipo
            if tipo_fer == "lazer":
                mult_fer = cfg["feriado_lazer"]
            elif tipo_fer == "fechado_parcial":
                mult_fer = 0.45
            elif tipo_fer == "carnaval":
                mult_fer = 0.75
            elif tipo_fer == "sexta_santa":
                mult_fer = 0.85
            else:
                mult_fer = 1.0

            # datas comemorativas e confraternizações
            mult_esp = ESPECIAIS[especial][prato] if especial else 1.0
            if (dia.month == 12 and dia.day <= 23 and dow < 5
                    and prato in PRATOS_CONFRATERNIZACAO):
                mult_esp *= 1.08

            # promoção: lift variável, canibalização e queda pós-promoção
            if prato in promo_hoje:
                mult_promo = 1 + (cfg["promo_sens"] - 1) * rng.uniform(0.7, 1.3)
            elif promo_hoje:
                mult_promo = 0.96
            elif (dia - timedelta(days=1), prato) in promos or (dia - timedelta(days=2), prato) in promos:
                mult_promo = 0.93
            else:
                mult_promo = 1.0

            mult_regime = 1.0
            if prato in REGIME_FINAL:
                mult_regime = 1 + (REGIME_FINAL[prato] - 1) * w_regime

            mu = (
                cfg["base"]
                * mult_dow * mult_mes * mult_tend * mult_clima * mult_chuva
                * mult_fer * mult_esp * mult_promo * mult_regime
                * f_comum
                * float(np.exp(idio[prato][i]))
                * float(np.exp(popular[prato][i]))
            )

            # eventos pontuais (grupos, aniversários)
            if rng.random() < P_EVENTO:
                mu *= rng.uniform(1.4, 2.2)

            # contagem sobredispersa (Poisson-Gama)
            mu = max(mu, 0.01)
            qtd = int(rng.negative_binomial(DISPERSAO_NB, DISPERSAO_NB / (DISPERSAO_NB + mu)))

            # ruptura: acabou o prato, venda menor que a demanda
            if rng.random() < P_RUPTURA:
                qtd = int(qtd * rng.uniform(0.55, 0.90))

            linhas.append({
                "data": dia,
                "prato": prato,
                "quantidade": max(0, qtd),
                "feriado": int(eh_feriado),
                "fim_de_semana": fds,
                "temperatura": temp,
                "promocao": int(prato in promo_hoje),
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