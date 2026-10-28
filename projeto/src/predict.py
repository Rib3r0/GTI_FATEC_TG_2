import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib
import numpy as np
import pandas as pd
from datetime import date, timedelta
from src.config import MODELS_DIR
from src.data_loader import carregar_vendas
from src.data_generator import feriados_no_intervalo


def carregar_pacote():
    return joblib.load(MODELS_DIR / "modelos_demanda.pkl")


def _linha_features(prato, dia, hist, feriados_set, data_inicio, temp_media):
    """Monta uma linha de features para o dia `dia` e prato `prato`."""
    ult = hist[prato]
    if len(ult) < 30:
        return None
    return {
        "prato": prato,
        "data": pd.Timestamp(dia),
        "dia_semana": dia.weekday(),
        "mes": dia.month,
        "dia_ano": dia.timetuple().tm_yday,
        "fim_de_semana": int(dia.weekday() >= 5),
        "feriado": int(dia in feriados_set),
        "temperatura": float(temp_media),
        "promocao": 0,
        "lag_1":  ult[-1],
        "lag_2":  ult[-2],
        "lag_7":  ult[-7],
        "lag_14": ult[-14],
        "mm_7":   float(np.mean(ult[-7:])),
        "mm_14":  float(np.mean(ult[-14:])),
        "mm_30":  float(np.mean(ult[-30:])),
        "std_7":  float(np.std(ult[-7:])),
        "dias_desde_inicio": (pd.Timestamp(dia) - data_inicio).days,
    }


def _predizer_ensemble(pacote, X: pd.DataFrame) -> np.ndarray:
    """Média ponderada dos modelos disponíveis."""
    feats = pacote["features"]
    preds = {}

    preds["ridge"] = pacote["ridge"].predict(X[feats])

    X_rf = X.copy()
    X_rf["prato"] = X_rf["prato"].cat.codes
    preds["random_forest"] = pacote["random_forest"].predict(X_rf[feats])

    preds["lightgbm"] = pacote["lightgbm"].predict(X[feats])

    if pacote.get("prophet"):
        yhat = []
        for _, row in X.iterrows():
            m = pacote["prophet"][row["prato"]]
            f = pd.DataFrame({"ds": [row["data"]]})
            yhat.append(m.predict(f)["yhat"].values[0])
        preds["prophet"] = np.array(yhat)

    pesos = pacote["pesos_ensemble"]
    pesos = {k: v for k, v in pesos.items() if k in preds}
    total = sum(pesos.values())
    pesos = {k: v / total for k, v in pesos.items()}
    return sum(pesos[k] * preds[k] for k in preds)


def _resolver_inicio(data_inicio_previsao=None) -> date:
    """Data em que a previsão começa.

    Prioridade: argumento > variável de ambiente DATA_INICIO_PREVISAO
    (definida pelo app ao rodar o pipeline) > hoje.
    """
    if data_inicio_previsao is None:
        data_inicio_previsao = os.environ.get("DATA_INICIO_PREVISAO") or date.today()
    return pd.Timestamp(data_inicio_previsao).date()


def prever_proximos_dias(n_dias: int = 7, data_inicio_previsao=None) -> pd.DataFrame:
    """Prevê `n_dias` a partir de `data_inicio_previsao` (padrão: hoje).

    O modelo só enxerga vendas anteriores à data de início. Se a última venda
    registrada for mais antiga que o dia anterior ao início, os dias dessa
    lacuna são estimados pelo próprio modelo (recursivamente) e não aparecem
    na saída. Se o início cair dentro do período com vendas reais, o resultado
    é o que o modelo teria previsto naquela data (útil para comparar com o real).
    """
    colunas = ["data", "prato", "previsao"]
    pacote = carregar_pacote()
    vendas = carregar_vendas()

    inicio = _resolver_inicio(data_inicio_previsao)
    fim = inicio + timedelta(days=n_dias - 1)

    # referências calculadas com todo o histórico (iguais às do treino)
    data_inicio = vendas["data"].min()
    categorias = vendas["prato"].astype("category").cat.categories

    # o modelo só pode usar o que veio antes da data de início
    vendas = vendas[vendas["data"].dt.date < inicio]
    if vendas.empty:
        return pd.DataFrame(columns=colunas)

    # Temperatura média dos últimos 30 dias (proxy para o futuro)
    if "temperatura" in vendas.columns:
        temp_media = float(
            vendas.sort_values("data").tail(30)["temperatura"].mean()
        )
    else:
        temp_media = 22.0

    hist = {p: sub.sort_values("data")["quantidade"].tolist()
            for p, sub in vendas.groupby("prato", observed=True)}

    # começa no dia seguinte à última venda; dias antes de `inicio` só alimentam o histórico
    primeiro_dia = vendas["data"].max().date() + timedelta(days=1)
    feriados_set = feriados_no_intervalo(primeiro_dia, fim)

    saida = []
    dia = primeiro_dia
    while dia <= fim:
        linhas = []
        for prato in hist:
            linha = _linha_features(prato, dia, hist, feriados_set,
                                     data_inicio, temp_media)
            if linha is not None:
                linhas.append(linha)
        if not linhas:
            break

        X = pd.DataFrame(linhas)
        X["prato"] = pd.Categorical(X["prato"], categories=categorias)

        preds = np.clip(np.round(_predizer_ensemble(pacote, X)), 0, None).astype(int)
        for i, p in enumerate(X["prato"]):
            hist[p].append(int(preds[i]))
            if dia >= inicio:
                saida.append({
                    "data": pd.Timestamp(dia),
                    "prato": p,
                    "previsao": int(preds[i]),
                })
        dia += timedelta(days=1)

    if not saida:
        return pd.DataFrame(columns=colunas)
    return pd.DataFrame(saida).sort_values(["data", "prato"]).reset_index(drop=True)


if __name__ == "__main__":
    print(prever_proximos_dias(7).to_string(index=False))