import pandas as pd

FEATURES_NUM = [
    "dia_semana", "mes", "dia_ano",
    "fim_de_semana", "feriado",
    "temperatura", "promocao",
    "lag_1", "lag_2", "lag_7", "lag_14",
    "mm_7", "mm_14", "mm_30",
    "std_7",
]
FEATURES_CAT = ["prato"]
FEATURES_ALL = FEATURES_CAT + FEATURES_NUM


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["prato", "data"]).reset_index(drop=True).copy()

    df["dia_semana"] = df["data"].dt.weekday
    df["mes"] = df["data"].dt.month
    df["dia_ano"] = df["data"].dt.dayofyear

    g = df.groupby("prato")["quantidade"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["mm_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["mm_14"] = g.transform(lambda s: s.shift(1).rolling(14).mean())
    df["mm_30"] = g.transform(lambda s: s.shift(1).rolling(30).mean())
    df["std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())

    # garante que temperatura e promoção existem (compatibilidade)
    if "temperatura" not in df.columns:
        df["temperatura"] = 22.0
    if "promocao" not in df.columns:
        df["promocao"] = 0

    df["prato"] = df["prato"].astype("category")
    return df


def split_temporal(df: pd.DataFrame, dias_val: int = 30):
    """Split temporal: últimos `dias_val` dias como validação."""
    corte = df["data"].max() - pd.Timedelta(days=dias_val - 1)
    treino = df[df["data"] < corte].dropna(subset=FEATURES_NUM).reset_index(drop=True)
    val = df[df["data"] >= corte].dropna(subset=FEATURES_NUM).reset_index(drop=True)
    return treino, val