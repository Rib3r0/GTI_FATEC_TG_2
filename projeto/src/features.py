import pandas as pd

FEATURES_NUM = [
    "dia_semana", "mes", "dia_ano","dias_desde_inicio",
    "fim_de_semana", "feriado",
    "lag_1", "lag_7", "lag_14",
    "mm_7", "mm_30",
]
FEATURES_CAT = ["prato"]
FEATURES_ALL = FEATURES_CAT + FEATURES_NUM


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recebe DataFrame long (data, prato, quantidade, feriado, fim_de_semana)
    e devolve o mesmo DataFrame enriquecido com features."""
    df = df.sort_values(["prato", "data"]).reset_index(drop=True).copy()

    # Calendário
    df["dia_semana"] = df["data"].dt.weekday
    df["mes"] = df["data"].dt.month
    df["dia_ano"] = df["data"].dt.dayofyear
    inicio = df["data"].min()
    df["dias_desde_inicio"] = (df["data"] - inicio).dt.days

    # Lags por prato
    g = df.groupby("prato")["quantidade"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)

    # Médias móveis (sem vazamento: shift antes de rolar)
    df["mm_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["mm_30"] = g.transform(lambda s: s.shift(1).rolling(30).mean())

    # Categórica
    df["prato"] = df["prato"].astype("category")

    return df


def split_temporal(df: pd.DataFrame, dias_val: int = 30):
    """Split temporal: últimos `dias_val` dias como validação."""
    corte = df["data"].max() - pd.Timedelta(days=dias_val - 1)
    treino = df[df["data"] < corte].dropna(subset=FEATURES_NUM).reset_index(drop=True)
    val = df[df["data"] >= corte].dropna(subset=FEATURES_NUM).reset_index(drop=True)
    return treino, val