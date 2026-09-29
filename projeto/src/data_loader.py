import pandas as pd
from src.config import RAW_DIR

def carregar_vendas() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "vendas.csv", parse_dates=["data"])
    return df.sort_values("data").reset_index(drop=True)

def carregar_pratos() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "pratos.csv")

def carregar_insumos() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "insumos.csv")

def carregar_ficha_tecnica() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "ficha_tecnica.csv")