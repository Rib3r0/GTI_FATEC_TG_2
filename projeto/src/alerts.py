import pandas as pd


LIMITE_EXCESSO = 0.5   # se sobrar >50% do estoque após o período → excesso
MARGEM_SEGURANCA = 1.10  # comprar 10% a mais que o calculado


def classificar(row: pd.Series, limiar_excesso: float = LIMITE_EXCESSO) -> str:
    """Classifica cada insumo em FALTA / EXCESSO / OK."""
    if row["saldo"] < 0:
        return "FALTA"
    if row["estoque_atual"] > 0 and (row["saldo"] / row["estoque_atual"]) > limiar_excesso:
        return "EXCESSO"
    return "OK"


def gerar_alertas(comparacao: pd.DataFrame) -> pd.DataFrame:
    """Aplica a classificação e ordena por prioridade (FALTA > EXCESSO > OK)."""
    df = comparacao.copy()
    df["status"] = df.apply(classificar, axis=1)
    ordem = {"FALTA": 0, "EXCESSO": 1, "OK": 2}
    df["_ordem"] = df["status"].map(ordem)
    return df.sort_values(["_ordem", "insumo"]).drop(columns="_ordem").reset_index(drop=True)


def sugestao_compra(alertas: pd.DataFrame,
                    margem: float = MARGEM_SEGURANCA) -> pd.DataFrame:
    """Para cada insumo em falta, sugere quantidade a comprar (com margem)."""
    falta = alertas[alertas["status"] == "FALTA"].copy()
    if falta.empty:
        return falta
    falta["qtd_sugerida"] = (-falta["saldo"] * margem).round(2)
    falta["custo_estimado"] = (falta["qtd_sugerida"] * falta["custo_unitario"]).round(2)
    return falta[["insumo", "unidade", "qtd_sugerida",
                  "custo_unitario", "custo_estimado"]].reset_index(drop=True)


def resumo_alertas(alertas: pd.DataFrame) -> dict:
    """Números-chave para o dashboard e relatório."""
    n_falta = int((alertas["status"] == "FALTA").sum())
    n_excesso = int((alertas["status"] == "EXCESSO").sum())

    custo_compra = float(
        (alertas.loc[alertas["status"] == "FALTA", "saldo"].abs()
         * alertas.loc[alertas["status"] == "FALTA", "custo_unitario"]).sum()
    )
    capital_em_risco = float(alertas["custo_excesso"].sum())

    return {
        "insumos_em_falta": n_falta,
        "insumos_com_excesso": n_excesso,
        "custo_estimado_compra_R$": round(custo_compra, 2),
        "capital_em_risco_R$": round(capital_em_risco, 2),
    }