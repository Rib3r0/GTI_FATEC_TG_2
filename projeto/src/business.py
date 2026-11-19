import os

import pandas as pd

from src.config import HORIZONTE_PREVISAO, PROCESSED_DIR
from src.data_loader import carregar_insumos, carregar_ficha_tecnica
from src.predict import prever_proximos_dias
from src.waste import (
    perdas_por_motivo, perdas_por_insumo, perdas_por_dia,
    taxa_desperdicio, balanco_insumos, consumo_teorico_vs_real,
)


def previsao_para_insumos(previsao: pd.DataFrame) -> pd.DataFrame:
    """Converte previsão de pratos em necessidade diária de insumos (kg/dia).

    Para cada linha (data, prato, previsao) multiplica pela quantidade de
    cada insumo na ficha técnica e soma por (data, insumo).
    Pratos sem ficha técnica são ignorados (não geram necessidade).
    """
    ficha = carregar_ficha_tecnica()
    if ficha.empty or previsao.empty:
        return pd.DataFrame(columns=["data", "insumo", "quantidade_necessaria"])

    # inner: só pratos que têm ficha; evita linhas com insumo NaN
    df = previsao.merge(ficha, on="prato", how="inner")
    df["quantidade_necessaria"] = df["previsao"] * df["quantidade_por_prato"]
    return (
        df.groupby(["data", "insumo"], as_index=False)["quantidade_necessaria"]
          .sum()
          .sort_values(["data", "insumo"])
          .reset_index(drop=True)
    )


def necessidade_total_periodo(necessidade_diaria: pd.DataFrame) -> pd.DataFrame:
    """Soma a necessidade de cada insumo no horizonte inteiro."""
    if necessidade_diaria.empty:
        return pd.DataFrame(columns=["insumo", "necessidade_total"])
    return (
        necessidade_diaria.groupby("insumo", as_index=False)["quantidade_necessaria"]
                          .sum()
                          .rename(columns={"quantidade_necessaria": "necessidade_total"})
    )


def comparar_estoque(necessidade_total: pd.DataFrame,
                     estoque: dict | None = None) -> pd.DataFrame:
    """Compara estoque atual com a necessidade prevista.

    Inclui **todos** os insumos cadastrados (não só os que aparecem na
    previsão). Insumo sem demanda no horizonte fica com necessidade 0.

    Devolve saldo (positivo = sobra, negativo = falta) e custos associados.
    """
    if estoque is None:
        from src.inventory import saldos_atuais
        estoque = saldos_atuais()

    insumos = carregar_insumos()
    if insumos.empty:
        return pd.DataFrame(columns=[
            "insumo", "unidade", "custo_unitario",
            "estoque_atual", "necessidade_total", "saldo",
            "custo_necessidade", "custo_excesso",
        ])

    # Parte de TODOS os insumos cadastrados e encaixa a necessidade (0 se não houver)
    df = insumos.rename(columns={"nome": "insumo"}).copy()
    if necessidade_total is None or necessidade_total.empty:
        df["necessidade_total"] = 0.0
    else:
        df = df.merge(necessidade_total, on="insumo", how="left")
        df["necessidade_total"] = df["necessidade_total"].fillna(0.0)

    df["estoque_atual"] = df["insumo"].map(estoque).fillna(0.0)
    df["saldo"] = df["estoque_atual"] - df["necessidade_total"]
    df["custo_unitario"] = df["custo_unitario"].fillna(0.0)

    df["custo_necessidade"] = df["necessidade_total"] * df["custo_unitario"]
    df["custo_excesso"] = df["saldo"].clip(lower=0) * df["custo_unitario"]

    return df[[
        "insumo", "unidade", "custo_unitario",
        "estoque_atual", "necessidade_total", "saldo",
        "custo_necessidade", "custo_excesso",
    ]].round(2)


def salvar_resultados(necessidade_diaria: pd.DataFrame,
                      comparacao: pd.DataFrame,
                      alertas: pd.DataFrame) -> None:
    """Persiste os artefatos da Parte 3 para uso no dashboard."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    necessidade_diaria.to_csv(PROCESSED_DIR / "necessidade_insumos_diaria.csv", index=False)
    comparacao.to_csv(PROCESSED_DIR / "comparacao_estoque.csv", index=False)
    alertas.to_csv(PROCESSED_DIR / "alertas.csv", index=False)


def recalcular_alertas(horizonte: int | None = None) -> pd.DataFrame:
    """Recalcula previsão → necessidade → comparação → alertas usando o estoque atual.

    Não reinicializa estoque nem simula vendas — seguro para chamar após
    cadastro, venda ou movimentação real.
    """
    horizonte = int(horizonte if horizonte is not None
                    else os.environ.get("HORIZONTE_PREVISAO", HORIZONTE_PREVISAO))

    previsao = prever_proximos_dias(horizonte)
    necessidade_diaria = previsao_para_insumos(previsao)
    necessidade_total = necessidade_total_periodo(necessidade_diaria)
    comparacao = comparar_estoque(necessidade_total)

    from src.alerts import gerar_alertas
    alertas = gerar_alertas(comparacao)
    salvar_resultados(necessidade_diaria, comparacao, alertas)
    return alertas


if __name__ == "__main__":
    # Só reinicializa / simula estoque quando o pipeline completo pede
    # (variável REGERRAR_ESTOQUE=1). No uso normal do dashboard (venda,
    # cadastro, aplicar horizonte) preserva o estoque real.
    regerar = os.environ.get("REGERAR_ESTOQUE", "").strip().lower() in (
        "1", "true", "sim", "yes",
    )
    if regerar:
        from src.inventory import inicializar_estoque, simular_vendas_recentes
        print("REGERRAR_ESTOQUE=1 → inicializando estoque e simulando vendas...")
        inicializar_estoque()
        simular_vendas_recentes(7)
    else:
        print("Usando estoque atual (sem reinicializar / simular vendas).")

    horizonte = int(os.environ.get("HORIZONTE_PREVISAO", HORIZONTE_PREVISAO))
    print(f"Gerando previsão dos próximos {horizonte} dias...")
    previsao = prever_proximos_dias(horizonte)

    print("\nConvertendo previsão em necessidade de insumos...")
    necessidade_diaria = previsao_para_insumos(previsao)
    necessidade_total = necessidade_total_periodo(necessidade_diaria)

    print("\nComparando com estoque atual (todos os insumos cadastrados)...")
    comparacao = comparar_estoque(necessidade_total)
    print(comparacao.to_string(index=False))

    print("\nGerando alertas...")
    from src.alerts import gerar_alertas, sugestao_compra, resumo_alertas
    alertas = gerar_alertas(comparacao)
    print(alertas[["insumo", "estoque_atual", "necessidade_total",
                   "saldo", "status"]].round(2).to_string(index=False))

    print("\n=== Sugestão de compra ===")
    compra = sugestao_compra(alertas)
    print(compra.to_string(index=False) if not compra.empty
          else "Nenhuma compra necessária.")

    print("\n=== Resumo executivo ===")
    for k, v in resumo_alertas(alertas).items():
        print(f"  {k}: {v}")

    print("\n=== Análise de desperdício ===")
    kpis_desp = taxa_desperdicio()
    for k, v in kpis_desp.items():
        print(f"  {k}: {v}")

    PROCESSSED = PROCESSED_DIR
    PROCESSSED.mkdir(parents=True, exist_ok=True)
    perdas_por_motivo().to_csv(PROCESSSED / "desperdicio_por_motivo.csv", index=False)
    perdas_por_insumo().to_csv(PROCESSSED / "desperdicio_por_insumo.csv", index=False)
    perdas_por_dia().to_csv(PROCESSSED / "desperdicio_por_dia.csv", index=False)
    balanco_insumos().to_csv(PROCESSSED / "balanco_insumos.csv", index=False)
    consumo_teorico_vs_real().to_csv(PROCESSSED / "consumo_teorico_vs_real.csv", index=False)

    salvar_resultados(necessidade_diaria, comparacao, alertas)
    print("\n[OK] Artefatos salvos em data/processed/")
