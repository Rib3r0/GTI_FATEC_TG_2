import pandas as pd
from src.config import ESTOQUE_INICIAL
from src.data_loader import carregar_insumos, carregar_ficha_tecnica
from src.predict import prever_proximos_dias
from src.inventory import inicializar_estoque


def previsao_para_insumos(previsao: pd.DataFrame) -> pd.DataFrame:
    """Converte previsão de pratos em necessidade diária de insumos (kg/dia).

    Para cada linha (data, prato, previsao) multiplica pela quantidade de
    cada insumo na ficha técnica e soma por (data, insumo)."""
    ficha = carregar_ficha_tecnica()
    df = previsao.merge(ficha, on="prato", how="left")
    df["quantidade_necessaria"] = df["previsao"] * df["quantidade_por_prato"]
    return (
        df.groupby(["data", "insumo"], as_index=False)["quantidade_necessaria"]
          .sum()
          .sort_values(["data", "insumo"])
          .reset_index(drop=True)
    )


def necessidade_total_periodo(necessidade_diaria: pd.DataFrame) -> pd.DataFrame:
    """Soma a necessidade de cada insumo no horizonte inteiro (7 dias)."""
    return (
        necessidade_diaria.groupby("insumo", as_index=False)["quantidade_necessaria"]
                          .sum()
                          .rename(columns={"quantidade_necessaria": "necessidade_total"})
    )


def comparar_estoque(necessidade_total: pd.DataFrame,
                     estoque: dict | None = None) -> pd.DataFrame:
    """Compara estoque atual com a necessidade prevista.

    Devolve saldo (positivo = sobra, negativo = falta) e custos associados."""
    if estoque is None:
        from src.inventory import saldos_atuais
        estoque = saldos_atuais()

    insumos = carregar_insumos()
    df = necessidade_total.merge(
        insumos, left_on="insumo", right_on="nome", how="left"
    )
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
    from src.config import PROCESSED_DIR
    necessidade_diaria.to_csv(PROCESSED_DIR / "necessidade_insumos_diaria.csv", index=False)
    comparacao.to_csv(PROCESSED_DIR / "comparacao_estoque.csv", index=False)
    alertas.to_csv(PROCESSED_DIR / "alertas.csv", index=False)


if __name__ == "__main__":
    inicializar_estoque()  

    print("Gerando previsão dos próximos 7 dias...")
    previsao = prever_proximos_dias(7)

    print("\nConvertendo previsão em necessidade de insumos...")
    necessidade_diaria = previsao_para_insumos(previsao)
    necessidade_total = necessidade_total_periodo(necessidade_diaria)

    print("\nComparando com estoque atual...")
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

    salvar_resultados(necessidade_diaria, comparacao, alertas)
    print("\n[OK] Artefatos salvos em data/processed/")