import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import pandas as pd
from datetime import date
from src.config import RAW_DIR, ESTOQUE_INICIAL
from src.data_loader import carregar_ficha_tecnica, carregar_insumos

MOV_FILE = RAW_DIR / "movimentacoes_estoque.csv"

TIPOS_VALIDOS = {"ENTRADA", "SAIDA_VENDA", "AJUSTE", "PERDA"}

COLUNAS = ["id", "data", "tipo", "insumo", "quantidade",
           "custo_unitario", "origem", "observacao"]


# ------------------------------------------------------------------
# Leitura / escrita
# ------------------------------------------------------------------
def _carregar() -> pd.DataFrame:
    if not MOV_FILE.exists():
        return pd.DataFrame(columns=COLUNAS)
    return pd.read_csv(MOV_FILE, parse_dates=["data"])


def _proximo_id(df: pd.DataFrame) -> int:
    return int(df["id"].max()) + 1 if not df.empty else 1


def _salvar(df: pd.DataFrame) -> None:
    df.to_csv(MOV_FILE, index=False)


# ------------------------------------------------------------------
# Registro de movimentações
# ------------------------------------------------------------------
def registrar_movimentacao(insumo: str, tipo: str, quantidade: float,
                            custo_unitario: float | None = 0.0,
                            origem: str = "", observacao: str = "",
                            data: date | None = None,
                            permitir_negativo: bool = False) -> None:
    if tipo not in TIPOS_VALIDOS:
        raise ValueError(f"Tipo inválido: {tipo}")

    if quantidade is None:
        raise ValueError("Quantidade é obrigatória.")
    quantidade = float(quantidade)
    if quantidade == 0:
        raise ValueError("Quantidade não pode ser zero.")

    custo_unitario = float(custo_unitario) if custo_unitario is not None else 0.0

    if tipo == "ENTRADA":
        quantidade = abs(quantidade)
    elif tipo in {"SAIDA_VENDA", "PERDA"}:
        quantidade = -abs(quantidade)
        if not permitir_negativo:
            atual = saldo_atual(insumo)
            if atual + quantidade < 0:
                raise ValueError(
                    f"Saldo insuficiente de '{insumo}': "
                    f"disponível {atual:.2f}, tentativa de baixa {abs(quantidade):.2f}."
                )

    df = _carregar()
    ts = pd.Timestamp.now() if data is None else pd.Timestamp(data)
    nova = pd.DataFrame([{
        "id": _proximo_id(df),
        "data": ts,
        "tipo": tipo,
        "insumo": insumo,
        "quantidade": float(quantidade),
        "custo_unitario": custo_unitario,
        "origem": origem,
        "observacao": observacao,
    }])
    _salvar(pd.concat([df, nova], ignore_index=True))


def registrar_entrada(insumo: str, quantidade: float,
                       custo_unitario: float = 0.0,
                       origem: str = "fornecedor",
                       observacao: str = "") -> None:
    """Registra uma entrada e atualiza o custo unitário do insumo
    pela média ponderada móvel (padrão contábil):
        novo_custo = (custo_atual * saldo + custo_novo * qtd) / (saldo + qtd)
    Se o custo informado for 0, o custo atual é mantido."""
    quantidade = float(quantidade)
    custo_informado = float(custo_unitario) if custo_unitario else 0.0

    if custo_informado > 0:
        from src.catalog import listar_insumos, editar_insumo
        saldo = saldo_atual(insumo)
        df_ins = listar_insumos()
        mask = df_ins["nome"] == insumo
        if mask.any():
            custo_atual = float(df_ins.loc[mask, "custo_unitario"].iloc[0])
            unidade = df_ins.loc[mask, "unidade"].iloc[0]
            if saldo > 0:
                novo_custo = (
                    custo_atual * saldo + custo_informado * quantidade
                ) / (saldo + quantidade)
            else:
                novo_custo = custo_informado
            editar_insumo(insumo, unidade, round(novo_custo, 4))

    registrar_movimentacao(insumo, "ENTRADA", quantidade,
                            custo_informado, origem, observacao)


def registrar_perda(insumo: str, quantidade: float,
                     motivo: str = "", observacao: str = "") -> None:
    registrar_movimentacao(insumo, "PERDA", quantidade,
                            origem=motivo, observacao=observacao)


def registrar_ajuste(insumo: str, quantidade_contada: float,
                      observacao: str = "") -> None:
    """Compara contagem física com saldo do sistema e registra a diferença."""
    atual = saldo_atual(insumo)
    delta = quantidade_contada - atual
    if abs(delta) < 1e-6:
        return
    registrar_movimentacao(insumo, "AJUSTE", delta,
                            origem="auditoria", observacao=observacao)


def registrar_saida_por_venda(prato: str, quantidade_vendida: int,
                               data: date | None = None) -> None:
    """Expande uma venda em consumo de insumos via ficha técnica."""
    ficha = carregar_ficha_tecnica()
    for _, row in ficha[ficha["prato"] == prato].iterrows():
        qtd = row["quantidade_por_prato"] * quantidade_vendida
        if qtd > 0:
            registrar_movimentacao(row["insumo"], "SAIDA_VENDA", qtd,
                                    origem=f"venda:{prato}", data=data)


# ------------------------------------------------------------------
# Consultas
# ------------------------------------------------------------------
def saldo_atual(insumo: str) -> float:
    df = _carregar()
    if df.empty:
        return 0.0
    return float(df[df["insumo"] == insumo]["quantidade"].sum())


def saldos_atuais() -> dict[str, float]:
    df = _carregar()
    if df.empty:
        return {}
    return df.groupby("insumo")["quantidade"].sum().to_dict()


def historico_insumo(insumo: str) -> pd.DataFrame:
    df = _carregar()
    return df[df["insumo"] == insumo].sort_values("data").reset_index(drop=True)


def todas_movimentacoes() -> pd.DataFrame:
    return _carregar().sort_values(["data", "id"]).reset_index(drop=True)


# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------
def inicializar_estoque(forcar: bool = False) -> None:
    """Cria ENTRADAs iniciais a partir de ESTOQUE_INICIAL (só se não existir)."""
    if MOV_FILE.exists() and not forcar:
        return
    if forcar and MOV_FILE.exists():
        MOV_FILE.unlink()

    insumos_df = carregar_insumos().set_index("nome")
    for insumo, qtd in ESTOQUE_INICIAL.items():
        if insumo not in insumos_df.index:
            continue
        custo = float(insumos_df.loc[insumo, "custo_unitario"])
        registrar_movimentacao(
            insumo, "ENTRADA", qtd, custo_unitario=custo,
            origem="estoque_inicial",
            observacao="carga inicial do sistema",
        )
    print(f"[OK] Estoque inicial registrado para {len(ESTOQUE_INICIAL)} insumos.")

def simular_vendas_recentes(dias: int = 7) -> None:
    """Registra SAIDA_VENDA para os últimos N dias de vendas.csv.
    Roda uma vez; ignora se já houver saídas simuladas."""
    from src.data_loader import carregar_vendas, carregar_ficha_tecnica

    movs = _carregar()
    if not movs.empty and movs["origem"].astype(str).str.startswith("sim:").any():
        print("[SKIP] Vendas recentes já simuladas.")
        return

    vendas = carregar_vendas().sort_values("data")
    corte = vendas["data"].max() - pd.Timedelta(days=dias - 1)
    vendas = vendas[vendas["data"] >= corte]

    ficha = carregar_ficha_tecnica()
    next_id = _proximo_id(movs)
    novas = []

    for _, v in vendas.iterrows():
        prato = v["prato"]
        qtd = int(v["quantidade"])
        data_v = v["data"]
        for _, row in ficha[ficha["prato"] == prato].iterrows():
            q = row["quantidade_por_prato"] * qtd
            if q <= 0:
                continue
            novas.append({
                "id": next_id,
                "data": data_v,
                "tipo": "SAIDA_VENDA",
                "insumo": row["insumo"],
                "quantidade": -abs(q),
                "custo_unitario": 0.0,
                "origem": f"sim:venda:{prato}",
                "observacao": "",
            })
            next_id += 1

    if novas:
        _salvar(pd.concat([movs, pd.DataFrame(novas)], ignore_index=True))
        print(f"[OK] Simuladas {len(novas)} saídas dos últimos {dias} dias.")


def _consumo_simulado_por_insumo(dias: int) -> dict[str, float]:
    """Consumo total por insumo nas vendas dos últimos N dias."""
    from src.data_loader import carregar_vendas, carregar_ficha_tecnica

    vendas = carregar_vendas().sort_values("data")
    corte = vendas["data"].max() - pd.Timedelta(days=dias - 1)
    vendas = vendas[vendas["data"] >= corte]

    ficha = carregar_ficha_tecnica()
    v = vendas.merge(ficha, on="prato", how="left")
    v["consumo"] = v["quantidade"] * v["quantidade_por_prato"]
    return v.groupby("insumo")["consumo"].sum().to_dict()


def resetar_estoque(dias_simulados: int = 7) -> None:
    """Recalibra o livro-razão:
    a carga inicial = ESTOQUE_INICIAL + consumo previsto nos próximos N dias.
    Assim, após as vendas simuladas, o saldo final bate com ESTOQUE_INICIAL."""
    if MOV_FILE.exists():
        MOV_FILE.unlink()

    insumos_df = carregar_insumos().set_index("nome")
    consumo = _consumo_simulado_por_insumo(dias_simulados)

    for insumo, qtd_final in ESTOQUE_INICIAL.items():
        if insumo not in insumos_df.index:
            continue
        custo = float(insumos_df.loc[insumo, "custo_unitario"])
        consumo_periodo = float(consumo.get(insumo, 0.0))
        qtd_inicial = qtd_final + consumo_periodo

        registrar_movimentacao(
            insumo, "ENTRADA", qtd_inicial,
            custo_unitario=custo,
            origem="estoque_inicial",
            observacao=(
                f"carga calibrada: saldo alvo {qtd_final:.2f} + "
                f"{consumo_periodo:.2f} consumidos em {dias_simulados}d"
            ),
            permitir_negativo=True,
        )

    simular_vendas_recentes(dias_simulados)
    print(f"[OK] Estoque resetado e calibrado para {dias_simulados} dias.")

if __name__ == "__main__":
    inicializar_estoque()
    print("\nSaldos atuais:")
    for k, v in saldos_atuais().items():
        print(f"  {k:15s} {v:8.2f}")