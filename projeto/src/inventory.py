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
                            custo_unitario: float = 0.0,
                            origem: str = "", observacao: str = "",
                            data: date | None = None) -> None:
    if tipo not in TIPOS_VALIDOS:
        raise ValueError(f"Tipo inválido: {tipo}")
    if quantidade == 0:
        raise ValueError("Quantidade não pode ser zero.")

    if tipo == "ENTRADA":
        quantidade = abs(quantidade)
    elif tipo in {"SAIDA_VENDA", "PERDA"}:
        quantidade = -abs(quantidade)
    # AJUSTE mantém o sinal recebido

    df = _carregar()
    nova = pd.DataFrame([{
        "id": _proximo_id(df),
        "data": pd.Timestamp(data or date.today()),
        "tipo": tipo,
        "insumo": insumo,
        "quantidade": float(quantidade),
        "custo_unitario": float(custo_unitario),
        "origem": origem,
        "observacao": observacao,
    }])
    _salvar(pd.concat([df, nova], ignore_index=True))


def registrar_entrada(insumo: str, quantidade: float,
                       custo_unitario: float = 0.0,
                       origem: str = "fornecedor",
                       observacao: str = "") -> None:
    registrar_movimentacao(insumo, "ENTRADA", quantidade,
                            custo_unitario, origem, observacao)


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


if __name__ == "__main__":
    inicializar_estoque()
    print("\nSaldos atuais:")
    for k, v in saldos_atuais().items():
        print(f"  {k:15s} {v:8.2f}")