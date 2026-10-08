import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import pandas as pd
from datetime import date
from src.config import RAW_DIR
from src.data_generator import feriados_no_intervalo


INSUMOS_FILE = RAW_DIR / "insumos.csv"
PRATOS_FILE  = RAW_DIR / "pratos.csv"
FICHA_FILE   = RAW_DIR / "ficha_tecnica.csv"
VENDAS_FILE  = RAW_DIR / "vendas.csv"

UNIDADES_VALIDAS = {"kg", "g", "L", "mL", "un"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _ler(path, colunas):
    if not path.exists():
        return pd.DataFrame(columns=colunas)
    return pd.read_csv(path)


def _proximo_id(df):
    return int(df["id"].max()) + 1 if not df.empty else 1


# ------------------------------------------------------------------
# INSUMOS
# ------------------------------------------------------------------
def listar_insumos() -> pd.DataFrame:
    return _ler(INSUMOS_FILE, ["id", "nome", "unidade", "custo_unitario"])


def adicionar_insumo(nome: str, unidade: str, custo_unitario: float) -> None:
    nome = nome.strip().lower().replace(" ", "_")
    if not nome:
        raise ValueError("Nome do insumo não pode ser vazio.")
    if unidade not in UNIDADES_VALIDAS:
        raise ValueError(f"Unidade inválida. Use uma de: {sorted(UNIDADES_VALIDAS)}")
    if custo_unitario < 0:
        raise ValueError("Custo unitário não pode ser negativo.")

    df = listar_insumos()
    if nome in df["nome"].values:
        raise ValueError(f"Insumo '{nome}' já existe.")

    nova = pd.DataFrame([{
        "id": _proximo_id(df),
        "nome": nome,
        "unidade": unidade,
        "custo_unitario": float(custo_unitario),
    }])
    pd.concat([df, nova], ignore_index=True).to_csv(INSUMOS_FILE, index=False)


def editar_insumo(nome: str, unidade: str, custo_unitario: float) -> None:
    df = listar_insumos()
    if nome not in df["nome"].values:
        raise ValueError(f"Insumo '{nome}' não encontrado.")
    if unidade not in UNIDADES_VALIDAS:
        raise ValueError(f"Unidade inválida.")
    if custo_unitario < 0:
        raise ValueError("Custo unitário não pode ser negativo.")

    df.loc[df["nome"] == nome, "unidade"] = unidade
    df.loc[df["nome"] == nome, "custo_unitario"] = float(custo_unitario)
    df.to_csv(INSUMOS_FILE, index=False)


def remover_insumo(nome: str) -> None:
    ficha = _ler(FICHA_FILE, ["prato", "insumo", "quantidade_por_prato"])
    if nome in ficha["insumo"].values:
        pratos = ficha[ficha["insumo"] == nome]["prato"].tolist()
        raise ValueError(
            f"Insumo '{nome}' está em uso nas fichas técnicas de: "
            f"{', '.join(pratos)}. Remova da ficha antes."
        )
    df = listar_insumos()
    df = df[df["nome"] != nome]
    df.to_csv(INSUMOS_FILE, index=False)


# ------------------------------------------------------------------
# PRATOS
# ------------------------------------------------------------------
def listar_pratos() -> pd.DataFrame:
    return _ler(PRATOS_FILE, ["id", "nome"])


def adicionar_prato(nome: str) -> None:
    nome = nome.strip().lower().replace(" ", "_")
    if not nome:
        raise ValueError("Nome do prato não pode ser vazio.")
    df = listar_pratos()
    if nome in df["nome"].values:
        raise ValueError(f"Prato '{nome}' já existe.")
    nova = pd.DataFrame([{"id": _proximo_id(df), "nome": nome}])
    pd.concat([df, nova], ignore_index=True).to_csv(PRATOS_FILE, index=False)


def editar_prato(nome_antigo: str, nome_novo: str) -> None:
    nome_novo = nome_novo.strip().lower().replace(" ", "_")
    if not nome_novo:
        raise ValueError("Nome do prato não pode ser vazio.")
    df = listar_pratos()
    if nome_antigo not in df["nome"].values:
        raise ValueError(f"Prato '{nome_antigo}' não encontrado.")
    if nome_novo in df["nome"].values and nome_novo != nome_antigo:
        raise ValueError(f"Prato '{nome_novo}' já existe.")

    df.loc[df["nome"] == nome_antigo, "nome"] = nome_novo
    df.to_csv(PRATOS_FILE, index=False)

    # Propaga o nome para ficha técnica e vendas
    for path in [FICHA_FILE, VENDAS_FILE]:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "prato" in d.columns:
            d["prato"] = d["prato"].replace(nome_antigo, nome_novo)
            d.to_csv(path, index=False)


def remover_prato(nome: str) -> None:
    ficha = _ler(FICHA_FILE, ["prato", "insumo", "quantidade_por_prato"])
    if nome in ficha["prato"].values:
        raise ValueError(
            f"Prato '{nome}' tem ficha técnica. Remova os itens antes."
        )
    vendas = _ler(VENDAS_FILE, ["data", "prato", "quantidade"])
    if nome in vendas["prato"].values:
        raise ValueError(f"Prato '{nome}' tem vendas registradas. Não pode ser removido.")
    df = listar_pratos()
    df = df[df["nome"] != nome]
    df.to_csv(PRATOS_FILE, index=False)


# ------------------------------------------------------------------
# FICHA TÉCNICA
# ------------------------------------------------------------------
def listar_ficha() -> pd.DataFrame:
    return _ler(FICHA_FILE, ["prato", "insumo", "quantidade_por_prato"])


def ficha_do_prato(prato: str) -> pd.DataFrame:
    ficha = listar_ficha()
    return ficha[ficha["prato"] == prato].reset_index(drop=True)


def adicionar_item_ficha(prato: str, insumo: str, quantidade: float) -> None:
    if quantidade <= 0:
        raise ValueError("Quantidade por prato deve ser > 0.")
    pratos = listar_pratos()
    if prato not in pratos["nome"].values:
        raise ValueError(f"Prato '{prato}' não existe.")
    insumos = listar_insumos()
    if insumo not in insumos["nome"].values:
        raise ValueError(f"Insumo '{insumo}' não existe.")

    ficha = listar_ficha()
    existe = ((ficha["prato"] == prato) & (ficha["insumo"] == insumo)).any()
    if existe:
        ficha.loc[(ficha["prato"] == prato) & (ficha["insumo"] == insumo),
                  "quantidade_por_prato"] = float(quantidade)
    else:
        nova = pd.DataFrame([{
            "prato": prato,
            "insumo": insumo,
            "quantidade_por_prato": float(quantidade),
        }])
        ficha = pd.concat([ficha, nova], ignore_index=True)
    ficha.to_csv(FICHA_FILE, index=False)


def remover_item_ficha(prato: str, insumo: str) -> None:
    ficha = listar_ficha()
    ficha = ficha[~((ficha["prato"] == prato) & (ficha["insumo"] == insumo))]
    ficha.to_csv(FICHA_FILE, index=False)


# ------------------------------------------------------------------
# VENDAS
# ------------------------------------------------------------------
def listar_vendas() -> pd.DataFrame:
    return _ler(VENDAS_FILE, ["data", "prato", "quantidade", "feriado", "fim_de_semana"])


def registrar_venda(prato: str, quantidade: int, data: date | None = None) -> None:
    if quantidade <= 0:
        raise ValueError("Quantidade deve ser > 0.")
    pratos = listar_pratos()
    if prato not in pratos["nome"].values:
        raise ValueError(f"Prato '{prato}' não existe.")

    data = data or date.today()
    vendas = listar_vendas()
    vendas["data"] = pd.to_datetime(vendas["data"]).dt.date
    feriados = feriados_no_intervalo(data, data)
    feriado = int(data in feriados)
    fds = int(data.weekday() >= 5)

    mask = (vendas["data"] == data) & (vendas["prato"] == prato)
    if mask.any():
        vendas.loc[mask, "quantidade"] += int(quantidade)
    else:
        nova = pd.DataFrame([{
            "data": data,
            "prato": prato,
            "quantidade": int(quantidade),
            "feriado": feriado,
            "fim_de_semana": fds,
        }])
        vendas = pd.concat([vendas, nova], ignore_index=True)

    vendas.to_csv(VENDAS_FILE, index=False)


def remover_venda(prato: str, data: date) -> None:
    vendas = listar_vendas()
    vendas["data"] = pd.to_datetime(vendas["data"]).dt.date
    vendas = vendas[~((vendas["data"] == data) & (vendas["prato"] == prato))]
    vendas.to_csv(VENDAS_FILE, index=False)