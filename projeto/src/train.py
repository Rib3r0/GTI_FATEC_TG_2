import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from src.config import MODELS_DIR
from src.data_loader import carregar_vendas
from src.features import criar_features, split_temporal, FEATURES_ALL, FEATURES_NUM
from src.evaluate import resumo_metricas, metricas_por_prato


# ------------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------------
def baseline_media_movel(df_val: pd.DataFrame) -> np.ndarray:
    """Predição ingênua: média móvel dos últimos 7 dias (já é feature mm_7)."""
    return df_val["mm_7"].to_numpy()


# ------------------------------------------------------------------
# MODELOS
# ------------------------------------------------------------------
def treinar_ridge(df_treino: pd.DataFrame) -> Pipeline:
    """Regressão linear com one-hot do prato."""
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["prato"])],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    pipe.fit(df_treino[FEATURES_ALL], df_treino["quantidade"])
    return pipe


def treinar_random_forest(df_treino: pd.DataFrame) -> RandomForestRegressor:
    X = df_treino.copy()
    X["prato"] = X["prato"].cat.codes
    model = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    model.fit(X[FEATURES_ALL], df_treino["quantidade"])
    return model


def treinar_lgbm(df_treino: pd.DataFrame) -> LGBMRegressor:
    model = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
        random_state=42, verbose=-1,
    )
    model.fit(df_treino[FEATURES_ALL], df_treino["quantidade"],
              categorical_feature=["prato"])
    return model


# ------------------------------------------------------------------
# WRAPPER DE PREDIÇÃO UNIFORME
# ------------------------------------------------------------------
def prever(modelo, nome: str, df_val: pd.DataFrame, prophet_models=None) -> np.ndarray:
    if nome == "baseline":
        return baseline_media_movel(df_val)
    if nome == "random_forest":
        X = df_val.copy()
        X["prato"] = X["prato"].cat.codes
        return modelo.predict(X[FEATURES_ALL])
    return modelo.predict(df_val[FEATURES_ALL])


# ------------------------------------------------------------------
# ENSEMBLE
# ------------------------------------------------------------------
def media_simples(preds: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.vstack(list(preds.values())), axis=0)


def media_ponderada(preds: dict[str, np.ndarray],
                    wapes: dict[str, float]) -> np.ndarray:
    """Peso inversamente proporcional ao WAPE (menor erro = maior peso)."""
    pesos = {k: 1.0 / max(v, 1e-6) for k, v in wapes.items()}
    total = sum(pesos.values())
    pesos = {k: v / total for k, v in pesos.items()}
    return sum(pesos[k] * preds[k] for k in preds)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("Carregando vendas...")
    vendas = carregar_vendas()

    print("Criando features...")
    df = criar_features(vendas)

    treino, val = split_temporal(df, dias_val=30)
    print(f"Treino: {treino['data'].min()} a {treino['data'].max()} ({len(treino)})")
    print(f"Val:    {val['data'].min()} a {val['data'].max()} ({len(val)})")

    # --- Baseline
    preds = {"baseline": baseline_media_movel(val)}

    # --- Ridge
    print("Treinando Ridge...")
    ridge = treinar_ridge(treino)
    preds["ridge"] = prever(ridge, "ridge", val)

    # --- Random Forest
    print("Treinando Random Forest...")
    rf = treinar_random_forest(treino)
    preds["random_forest"] = prever(rf, "random_forest", val)

    # --- LightGBM
    print("Treinando LightGBM...")
    lgbm = treinar_lgbm(treino)
    preds["lightgbm"] = prever(lgbm, "lightgbm", val)

    # --- Métricas individuais
    y_true = val["quantidade"].to_numpy()
    linhas = []
    wapes = {}
    for nome, p in preds.items():
        p_clip = np.clip(p, 0, None)
        m = resumo_metricas(y_true, p_clip)
        m["modelo"] = nome
        linhas.append(m)
        wapes[nome] = m["WAPE (%)"]
    tabela = pd.DataFrame(linhas).set_index("modelo").round(3)
    print("\n=== Comparação de modelos ===")
    print(tabela)

    # --- Ensembles (excluindo baseline dos ensembles)
    base_models = {k: v for k, v in preds.items() if k != "baseline"}
    base_wapes = {k: wapes[k] for k in base_models}

    ens_simples = np.clip(media_simples(base_models), 0, None)
    ens_pond = np.clip(media_ponderada(base_models, base_wapes), 0, None)

    print("\n=== Ensembles ===")
    for nome, p in [("ensemble_media_simples", ens_simples),
                    ("ensemble_media_ponderada", ens_pond)]:
        m = resumo_metricas(y_true, p)
        m["modelo"] = nome
        print(f"{nome}: " + " | ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "modelo"))

    # --- WAPE por prato para o melhor modelo (ensemble ponderado)
    val_out = val.copy()
    val_out["previsao"] = ens_pond
    print("\n=== WAPE por prato (ensemble ponderado) ===")
    print(metricas_por_prato(val_out)[["MAE", "WAPE (%)", "Viés"]])

    # --- Escolha do melhor
    resultados = {**{k: wapes[k] for k in preds},
                  "ensemble_media_simples": resumo_metricas(y_true, ens_simples)["WAPE (%)"],
                  "ensemble_media_ponderada": resumo_metricas(y_true, ens_pond)["WAPE (%)"]}
    melhor_nome = min(resultados, key=resultados.get)
    print(f"\n>>> Melhor modelo por WAPE: {melhor_nome} ({resultados[melhor_nome]:.3f}%)")

    # --- Salva tudo
    pacote = {
        "ridge": ridge,
        "random_forest": rf,
        "lightgbm": lgbm,
        "features": FEATURES_ALL,
        "pesos_ensemble": {
            k: (1.0 / max(v, 1e-6)) for k, v in base_wapes.items()
        },
        "tabela_metricas": tabela,
        "melhor_modelo": melhor_nome,
    }
    caminho = MODELS_DIR / "modelos_demanda.pkl"
    joblib.dump(pacote, caminho)
    print(f"✔ Pacote salvo em {caminho}")


if __name__ == "__main__":
    main()