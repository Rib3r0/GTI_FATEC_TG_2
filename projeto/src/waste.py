import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import pandas as pd
from src.inventory import todas_movimentacoes
from src.data_loader import carregar_insumos, carregar_vendas, carregar_ficha_tecnica


def _movs_com_custo() -> pd.DataFrame:
    movs = todas_movimentacoes()
    if movs.empty:
        return movs
    insumos = carregar_insumos().set_index("nome")

    def _custo(row):
        if row["custo_unitario"] > 0:
            return float(row["custo_unitario"])
        if row["insumo"] in insumos.index:
            return float(insumos.loc[row["insumo"], "custo_unitario"])
        return 0.0

    movs = movs.copy()
    movs["custo_unitario"] = movs.apply(_custo, axis=1)
    movs["custo_total"] = movs["quantidade"].abs() * movs["custo_unitario"]
    return movs


# ------------------------------------------------------------------
# Perdas
# ------------------------------------------------------------------
def perdas_por_motivo() -> pd.DataFrame:
    movs = _movs_com_custo()
    perdas = movs[movs["tipo"] == "PERDA"]
    if perdas.empty:
        return pd.DataFrame(columns=["motivo", "quantidade", "custo"])
    return (
        perdas.groupby("origem")
        .agg(quantidade=("quantidade", lambda s: s.abs().sum()),
             custo=("custo_total", "sum"))
        .reset_index()
        .rename(columns={"origem": "motivo"})
        .sort_values("custo", ascending=False)
        .round(2)
    )


def perdas_por_insumo() -> pd.DataFrame:
    movs = _movs_com_custo()
    perdas = movs[movs["tipo"] == "PERDA"]
    if perdas.empty:
        return pd.DataFrame(columns=["insumo", "quantidade", "custo"])
    return (
        perdas.groupby("insumo")
        .agg(quantidade=("quantidade", lambda s: s.abs().sum()),
             custo=("custo_total", "sum"))
        .reset_index()
        .sort_values("custo", ascending=False)
        .round(2)
    )


def perdas_por_dia() -> pd.DataFrame:
    movs = _movs_com_custo()
    perdas = movs[movs["tipo"] == "PERDA"]
    if perdas.empty:
        return pd.DataFrame(columns=["data", "quantidade", "custo"])
    perdas = perdas.copy()
    perdas["data"] = perdas["data"].dt.date
    return (
        perdas.groupby("data")
        .agg(quantidade=("quantidade", lambda s: s.abs().sum()),
             custo=("custo_total", "sum"))
        .reset_index()
        .round(2)
    )


# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
def taxa_desperdicio() -> dict:
    movs = _movs_com_custo()
    entradas = movs[movs["tipo"] == "ENTRADA"]
    perdas = movs[movs["tipo"] == "PERDA"]

    custo_entrada = float(entradas["custo_total"].sum())
    custo_perda = float(perdas["custo_total"].sum())
    taxa = (custo_perda / custo_entrada * 100) if custo_entrada > 0 else 0.0

    return {
        "custo_total_entradas": round(custo_entrada, 2),
        "custo_total_perdas": round(custo_perda, 2),
        "taxa_desperdicio_pct": round(taxa, 2),
        "n_registros_perda": int(len(perdas)),
    }


# ------------------------------------------------------------------
# Balanço contábil por insumo
# ------------------------------------------------------------------
def balanco_insumos() -> pd.DataFrame:
    """Demonstra: comprado = consumido_vendas + perdido + saldo (± ajustes)."""
    movs = _movs_com_custo()
    if movs.empty:
        return pd.DataFrame()

    por_tipo = movs.groupby(["insumo", "tipo"])["quantidade"].sum().unstack(fill_value=0)

    zero = pd.Series(0.0, index=por_tipo.index)
    comprado = por_tipo.get("ENTRADA", zero)
    consumido = por_tipo.get("SAIDA_VENDA", zero).abs()
    perdido = por_tipo.get("PERDA", zero).abs()
    ajuste = por_tipo.get("AJUSTE", zero)
    saldo = comprado - consumido - perdido + ajuste

    df = pd.DataFrame({
        "comprado": comprado,
        "consumido_vendas": consumido,
        "perdido": perdido,
        "ajustes": ajuste,
        "saldo_atual": saldo,
    }).reset_index()

    df["pct_perdido"] = (df["perdido"] / df["comprado"].replace(0, pd.NA) * 100).fillna(0)
    return df.round(2).sort_values("pct_perdido", ascending=False)


# ------------------------------------------------------------------
# Consumo teórico vs. saída real
# ------------------------------------------------------------------
def consumo_teorico_vs_real() -> pd.DataFrame:
    """Teórico = vendas × ficha técnica.
    Real = movimentações de SAIDA_VENDA.
    Diferença = erro de porção / desperdício não registrado."""
    from src.inventory import todas_movimentacoes

    vendas = carregar_vendas()
    ficha = carregar_ficha_tecnica()
    movs = todas_movimentacoes()

    v = vendas.merge(ficha, on="prato", how="left")
    v["teorico"] = v["quantidade"] * v["quantidade_por_prato"]
    teorico = v.groupby("insumo")["teorico"].sum().reset_index()

    saidas = movs[movs["tipo"] == "SAIDA_VENDA"].copy()
    if saidas.empty:
        teorico["saida_real"] = 0.0
    else:
        real = saidas.groupby("insumo")["quantidade"].sum().abs().reset_index()
        real.columns = ["insumo", "saida_real"]
        teorico = teorico.merge(real, on="insumo", how="left").fillna({"saida_real": 0})

    teorico["diferenca"] = (teorico["saida_real"] - teorico["teorico"]).round(2)
    teorico["diferenca_pct"] = (
        teorico["diferenca"] / teorico["teorico"].replace(0, pd.NA) * 100
    ).fillna(0).round(2)
    return teorico.round(2)