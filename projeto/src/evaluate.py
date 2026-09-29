import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def vies(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def resumo_metricas(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "WAPE (%)": wape(y_true, y_pred),
        "Viés": vies(y_true, y_pred),
    }


def metricas_por_prato(df: pd.DataFrame, col_true="quantidade", col_pred="previsao") -> pd.DataFrame:
    linhas = []
    for prato, sub in df.groupby("prato", observed=True):
        m = resumo_metricas(sub[col_true], sub[col_pred])
        m["prato"] = prato
        m["n"] = len(sub)
        linhas.append(m)
    return pd.DataFrame(linhas).set_index("prato").round(3)