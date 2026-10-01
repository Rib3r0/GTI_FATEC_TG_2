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
def prever(modelo, nome: str, df_val: pd.DataFrame) -> np.ndarray:
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

    y_true = val["quantidade"].to_numpy()

    # ---------- Modelos base ----------
    preds: dict[str, np.ndarray] = {}

    print("Baseline (média móvel 7d)...")
    preds["baseline"] = baseline_media_movel(val)

    print("Treinando Ridge...")
    ridge = treinar_ridge(treino)
    preds["ridge"] = prever(ridge, "ridge", val)

    print("Treinando Random Forest...")
    rf = treinar_random_forest(treino)
    preds["random_forest"] = prever(rf, "random_forest", val)

    print("Treinando LightGBM...")
    lgbm = treinar_lgbm(treino)
    preds["lightgbm"] = prever(lgbm, "lightgbm", val)

    # ---------- Ensembles (calculados ANTES da tabela) ----------
    base_models = {k: v for k, v in preds.items() if k != "baseline"}
    wapes_base = {
        k: resumo_metricas(y_true, np.clip(v, 0, None))["WAPE (%)"]
        for k, v in base_models.items()
    }

    preds["ensemble_media_simples"] = np.clip(media_simples(base_models), 0, None)
    preds["ensemble_media_ponderada"] = np.clip(
        media_ponderada(base_models, wapes_base), 0, None
    )

    # ---------- Tabela única com TODOS os modelos ----------
    linhas = []
    for nome, p in preds.items():
        m = resumo_metricas(y_true, np.clip(p, 0, None))
        m["modelo"] = nome
        linhas.append(m)

    tabela = (
        pd.DataFrame(linhas)
        .set_index("modelo")
        .round(3)
        .sort_values("WAPE (%)")
    )
    print("\n=== Comparação de todos os modelos ===")
    print(tabela)

    # ---------- Escolha do melhor (excluindo baseline) ----------
    candidatos = tabela.drop(index="baseline", errors="ignore")
    melhor_nome = candidatos["WAPE (%)"].idxmin()
    print(f"\n>>> Modelo final escolhido: {melhor_nome} "
          f"(WAPE {tabela.loc[melhor_nome, 'WAPE (%)']:.3f}%)")

    # ---------- WAPE por prato com o melhor ----------
    val_out = val.copy()
    val_out["previsao"] = preds[melhor_nome]
    print("\n=== WAPE por prato (modelo final) ===")
    print(metricas_por_prato(val_out)[["MAE", "WAPE (%)", "Viés"]])

    # ---------- Salva pacote ----------
    pacote = {
        "ridge": ridge,
        "random_forest": rf,
        "lightgbm": lgbm,
        "features": FEATURES_ALL,
        "pesos_ensemble": {
            k: (1.0 / max(v, 1e-6)) for k, v in wapes_base.items()
        },
        "tabela_metricas": tabela,
        "melhor_modelo": melhor_nome,
    }
    caminho = MODELS_DIR / "modelos_demanda.pkl"
    joblib.dump(pacote, caminho)
    print(f"[OK] Pacote salvo em {caminho}")


if __name__ == "__main__":
    main()