import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.catalog import (
    adicionar_insumo, adicionar_item_ficha, adicionar_prato,
    editar_insumo, editar_prato, ficha_do_prato,
    listar_insumos, listar_pratos, listar_vendas,
    registrar_venda, remover_insumo, remover_item_ficha,
    remover_prato, remover_venda,
)
from src.config import HORIZONTE_PREVISAO, MODELS_DIR, PROCESSED_DIR
from src.data_loader import carregar_insumos, carregar_vendas
from src.inventory import (
    registrar_ajuste, registrar_entrada, registrar_perda,
    registrar_saida_por_venda, saldo_atual, saldos_atuais, todas_movimentacoes,
)
from src.waste import taxa_desperdicio

st.set_page_config(
    page_title="Previsão de Demanda — Restaurante",
    page_icon="🍽️",
    layout="wide",
)


# ------------------------------------------------------------------
# FORMATAÇÃO: números no padrão brasileiro e nomes legíveis
# (só afeta o que aparece na tela; os dados continuam com os nomes originais)
# ------------------------------------------------------------------
NOMES_ESPECIAIS = {"feijao": "Feijão"}

NOMES_MODELOS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "prophet": "Prophet",
    "ensemble_media_ponderada": "Ensemble (média ponderada)",
}

ROTULOS_VALORES = {
    "ENTRADA": "Entrada", "SAIDA": "Saída", "SAIDA_VENDA": "Saída (venda)",
    "PERDA": "Perda", "AJUSTE": "Ajuste",
    "FALTA": "Falta", "OK": "OK", "EXCESSO": "Excesso",
    "validade": "Validade", "preparo": "Preparo", "sobra": "Sobra",
    "avaria": "Avaria", "outro": "Outro",
}

COLUNAS = {
    "data": "Data", "prato": "Prato", "insumo": "Insumo", "nome": "Nome",
    "unidade": "Unidade", "quantidade": "Quantidade", "previsao": "Previsão",
    "total": "Total", "feriado": "Feriado", "fim_de_semana": "Fim de semana",
    "temperatura": "Temperatura (°C)", "promocao": "Promoção", "tipo": "Tipo",
    "status": "Status", "motivo": "Motivo", "origem": "Origem",
    "obs": "Observação", "observacao": "Observação",
    "custo_unitario": "Custo unitário", "custo": "Custo",
    "custo_total": "Custo total", "custo_excesso": "Custo do excesso",
    "custo_estimado": "Custo estimado", "valor_total": "Valor em estoque",
    "saldo_atual": "Saldo atual", "saldo": "Saldo", "estoque_atual": "Estoque atual",
    "necessidade_total": "Necessidade", "quantidade_necessaria": "Quantidade necessária",
    "qtd_sugerida": "Qtd. sugerida", "quantidade_por_prato": "Qtd. por prato",
    "saldo_pos": "Saldo após a venda", "ok": "Situação",
    "teorico": "Consumo teórico", "saida_real": "Consumo real",
    "dia_semana": "Dia da semana", "dia_ano": "Dia do ano", "mes": "Mês", "ano": "Ano",
}

ROTULOS_GRAFICO = {
    "data": "Data", "quantidade": "Quantidade", "prato": "Prato", "insumo": "Insumo",
    "previsao": "Previsão", "total": "Total", "tipo": "Tipo", "tipo_dia": "Tipo de dia",
    "dia_semana_pt": "Dia da semana", "custo": "Custo (R$)",
    "custo_estimado": "Custo estimado (R$)", "quantidade_necessaria": "Quantidade (kg)",
    "motivo": "Motivo", "value": "Quantidade (kg)", "variable": "Tipo",
}

_COL_NOMES = ("prato", "insumo", "nome")
_COL_ROTULOS = ("tipo", "status", "motivo")
_COL_SIMNAO = ("feriado", "fim_de_semana", "promocao")
_COL_INT_MILHAR = ("quantidade", "total", "previsao")


def _vazio_na(x) -> bool:
    return x is None or (not isinstance(x, str) and bool(pd.isna(x)))


def nome_bonito(txt):
    """'frango_grelhado' -> 'Frango Grelhado'; 'feijao' -> 'Feijão'."""
    if _vazio_na(txt):
        return txt
    s = str(txt)
    if s in NOMES_ESPECIAIS:
        return NOMES_ESPECIAIS[s]
    if "_" in s or s == s.lower():
        return s.replace("_", " ").title()
    return s  # já está formatado (ex.: cadastrado pelo usuário com maiúsculas/acentos)


def rotulo(txt):
    """'SAIDA_VENDA' -> 'Saída (venda)'; 'validade' -> 'Validade'."""
    if _vazio_na(txt):
        return txt
    s = str(txt)
    if s in ROTULOS_VALORES:
        return ROTULOS_VALORES[s]
    if s == s.lower() or s == s.upper():
        return s.replace("_", " ").capitalize()
    return s


def titulo_coluna(c) -> str:
    if c in COLUNAS:
        return COLUNAS[c]
    s = str(c)
    return s.replace("_", " ").capitalize() if s == s.lower() else s


def _br(txt: str) -> str:
    """Troca separadores do padrão americano para o brasileiro (1,234.50 -> 1.234,50)."""
    return txt.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_num(x, casas: int = 2) -> str:
    if _vazio_na(x):
        return "—"
    return _br(f"{float(x):,.{casas}f}")


def fmt_int(x) -> str:
    return fmt_num(x, 0)


def fmt_brl(x, casas: int = 2) -> str:
    if _vazio_na(x):
        return "—"
    sinal = "-" if float(x) < 0 else ""
    return f"{sinal}R$ {fmt_num(abs(float(x)), casas)}"


def fmt_pct(x, casas: int = 1) -> str:
    return "—" if _vazio_na(x) else f"{fmt_num(x, casas)}%"


def fmt_qtd(x, unidade: str = "", casas: int = 2) -> str:
    return f"{fmt_num(x, casas)} {unidade}".strip()


def _mk_num(casas):
    return lambda x: fmt_num(x, casas)


def _mk_data(formato):
    return lambda x: "—" if _vazio_na(x) else x.strftime(formato)


def _eh_dinheiro(c) -> bool:
    c = str(c).lower()
    return any(k in c for k in ("custo", "valor", "capital", "preco", "preço")) and not c.startswith("n_")


def _eh_pct(c) -> bool:
    c = str(c).lower()
    return "pct" in c or "%" in c


def _formatar_fmt_qualquer(x) -> str:
    try:
        return fmt_num(x, 2)
    except (TypeError, ValueError):
        return str(x)


def mostrar_tabela(df, *, indice=False, pivot=False, formato=None, casas=None,
                   titulos=None, destaque=None, **kw):
    """Mostra um DataFrame com nomes e números formatados (padrão brasileiro).

    indice   : mostra o índice (use em tabelas dinâmicas / agregadas)
    pivot    : colunas são nomes de itens (pratos/insumos), não campos
    formato  : função de formatação única para as células numéricas (pivot)
    casas    : {coluna: casas decimais} para sobrescrever o padrão
    titulos  : {coluna: título} para sobrescrever o título automático
    destaque : (função_de_estilo, [colunas]) para colorir células
    """
    d = df.copy()
    casas = casas or {}
    titulos = titulos or {}

    # ---- valores: nomes de itens, rótulos, sim/não, datas em texto
    if not pivot:
        for c in list(d.columns):
            if c in _COL_NOMES:
                d[c] = d[c].astype(object).map(nome_bonito)
            elif c in _COL_ROTULOS:
                d[c] = d[c].astype(object).map(rotulo)
            elif c in _COL_SIMNAO and set(d[c].dropna().unique()) <= {0, 1}:
                d[c] = d[c].map({0: "Não", 1: "Sim"})
            elif str(c) == "data" or str(c).startswith("data_"):
                s = d[c].dropna()
                if (not pd.api.types.is_datetime64_any_dtype(d[c]) and len(s)
                        and not isinstance(s.iloc[0], date)):
                    conv = pd.to_datetime(d[c], errors="coerce")
                    if conv.notna().all():
                        d[c] = conv

    # ---- formatadores por coluna (chaves = nomes técnicos)
    fm = {}
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            com_hora = bool((s.dt.hour.fillna(0) != 0).any() or (s.dt.minute.fillna(0) != 0).any())
            fm[c] = _mk_data("%d/%m/%Y %H:%M" if com_hora else "%d/%m/%Y")
        elif s.dtype == object and len(s.dropna()) and isinstance(s.dropna().iloc[0], date):
            fm[c] = _mk_data("%d/%m/%Y")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if formato is not None:
                fm[c] = formato
            elif c in casas:
                fm[c] = _mk_num(casas[c])
            elif _eh_dinheiro(c):
                fm[c] = fmt_brl
            elif _eh_pct(c):
                fm[c] = lambda x: fmt_pct(x, 2)
            elif c == "temperatura":
                fm[c] = _mk_num(1)
            elif pd.api.types.is_integer_dtype(s):
                fm[c] = fmt_int if c in _COL_INT_MILHAR else (lambda x: str(int(x)))
            else:
                fm[c] = _mk_num(3 if c == "quantidade_por_prato" else 2)

    # ---- títulos
    if pivot:
        nome_cols = d.columns.name
        d.columns = [nome_bonito(c) for c in d.columns]
        fm = {nome_bonito(k): v for k, v in fm.items()}
        if nome_cols:
            d.columns.name = titulo_coluna(nome_cols)
        ren = {}
    else:
        ren = {c: titulos.get(c, titulo_coluna(c)) for c in d.columns}
        d = d.rename(columns=ren)
        fm = {ren[k]: v for k, v in fm.items()}

    # ---- índice
    idx_fmt = None
    if indice:
        nome_idx = d.index.name
        if nome_idx in _COL_NOMES:
            d.index = d.index.map(nome_bonito)
        if pd.api.types.is_datetime64_any_dtype(d.index):
            idx_fmt = _mk_data("%d/%m/%Y")
        elif d.index.dtype == object and len(d.index) and isinstance(d.index[0], date):
            idx_fmt = _mk_data("%d/%m/%Y")
        if nome_idx:
            d.index.name = titulo_coluna(nome_idx)

    if d.size > 150_000:  # Styler fica pesado em tabelas enormes: mostra sem formatar
        return st.dataframe(d, use_container_width=True, hide_index=not indice, **kw)

    sty = d.style.format(fm, na_rep="—") if fm else d.style
    if idx_fmt is not None:
        sty = sty.format_index(idx_fmt, axis=0)
    if destaque:
        func, cols = destaque
        sty = sty.map(func, subset=[ren.get(c, c) for c in cols])
    return st.dataframe(sty, use_container_width=True, hide_index=not indice, **kw)


def bonito(df, extras=()):
    """Cópia do DataFrame com nomes de pratos/insumos formatados (para gráficos)."""
    d = df.copy()
    for c in list(_COL_NOMES) + list(extras):
        if c in d.columns:
            d[c] = d[c].astype(object).map(nome_bonito)
    return d


def _x_eh_data(fig) -> bool:
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is not None and len(x):
            return bool(np.issubdtype(np.asarray(x).dtype, np.datetime64)
                        or isinstance(x[0], (pd.Timestamp, date)))
    return False


def grafico(fig):
    """Mostra um gráfico Plotly com separadores brasileiros e datas dd/mm/aaaa."""
    fig.update_layout(separators=",.")
    if _x_eh_data(fig):
        fig.update_xaxes(hoverformat="%d/%m/%Y")
        if fig.layout.xaxis.tickformat is None:
            fig.update_xaxes(tickformat="%d/%m/%y")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------------
def filtrar_vendas(df: pd.DataFrame, data_ini, data_fim, pratos_sel) -> pd.DataFrame:
    """Aplica os filtros globais de período e pratos."""
    if df.empty:
        return df
    mask = (
        (df["data"].dt.date >= data_ini)
        & (df["data"].dt.date <= data_fim)
        & (df["prato"].isin(pratos_sel))
    )
    return df[mask].copy()


def filtrar_por_insumo(df: pd.DataFrame, insumos_sel, col: str = "insumo") -> pd.DataFrame:
    """Aplica o filtro global de insumos."""
    if df.empty or col not in df.columns:
        return df
    return df[df[col].isin(insumos_sel)].copy()


def _flash(tipo: str, msg: str):
    """Guarda uma mensagem para aparecer (como aviso) depois do próximo rerun."""
    st.session_state.setdefault("flash", []).append((tipo, msg))


def _env_utf8(horizonte: int | None = None, inicio: date | None = None) -> dict:
    """Ambiente dos subprocessos. Sem argumentos, usa o horizonte e a data de
    início escolhidos na barra lateral, para o pipeline ficar coerente com a tela."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    horizonte = horizonte if horizonte is not None else st.session_state.get("horizonte_atual")
    if horizonte is not None:
        env["HORIZONTE_PREVISAO"] = str(horizonte)
    inicio = inicio if inicio is not None else st.session_state.get("inicio_previsao")
    if inicio is not None:
        env["DATA_INICIO_PREVISAO"] = pd.Timestamp(inicio).date().isoformat()
    return env


def _rodar_modulos(modulos: list[str], env: dict | None = None) -> tuple[bool, str]:
    """Roda módulos `src.*` em sequência. Retorna (deu_certo, mensagem_de_erro)."""
    for mod in modulos:
        r = subprocess.run(
            [sys.executable, "-m", mod],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        if r.returncode != 0:
            return False, f"Erro em {mod}:\n{(r.stderr or '')[-500:]}"
    return True, ""


def _limpar_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def _reprocessar(horizonte: int | None = None, inicio: date | None = None) -> bool:
    """Recalcula previsões e alertas (src.business) e limpa o cache."""
    ok, erro = _rodar_modulos(["src.business"], env=_env_utf8(horizonte, inicio))
    _limpar_cache()
    if not ok:
        st.error(erro)
    return ok


def _executar_pipeline(modulos: list[str], rotulo: str):
    """Roda uma sequência de módulos mostrando progresso; para o app se algo falhar."""
    with st.spinner("Gerando dados, treinando e calculando alertas..."):
        ok, erro = _rodar_modulos(modulos, env=_env_utf8())
    _limpar_cache()
    if not ok:
        st.sidebar.error(erro)
        st.stop()
    _flash("success", f"{rotulo} concluído.")
    st.rerun()


# ------------------------------------------------------------------
# CACHE
# ------------------------------------------------------------------
@st.cache_data
def load_vendas():
    return carregar_vendas()


@st.cache_data
def load_csv(nome: str) -> pd.DataFrame:
    p = PROCESSED_DIR / nome
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_resource
def load_modelo():
    import joblib
    p = MODELS_DIR / "modelos_demanda.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.dialog("Confirmar movimentação")
def dialog_movimentacao(tipo: str, campos: dict, confirmar_callback):
    st.write(f"**Tipo:** {tipo}")
    for k, v in campos.items():
        st.write(f"**{k}:** {v}")
    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("✅ Confirmar", type="primary", use_container_width=True):
        confirmar_callback()
        if _reprocessar():
            _flash("success", "Movimentação registrada.")
            st.rerun()
        else:
            st.warning("A movimentação foi registrada, mas o recálculo dos alertas falhou.")
    if c2.button("❌ Cancelar", use_container_width=True):
        st.rerun()


# Avisos pendentes (gerados antes de um rerun)
for _tipo, _msg in st.session_state.pop("flash", []):
    st.toast(_msg, icon="✅" if _tipo == "success" else "⚠️")


# ------------------------------------------------------------------
# BARRA LATERAL
# ------------------------------------------------------------------
st.sidebar.header("Filtros")

# --- Período (com atalhos logo abaixo)
vendas_full = load_vendas()

if not vendas_full.empty:
    data_min = vendas_full["data"].min().date()
    data_max = vendas_full["data"].max().date()

    # Contador de versão: muda a chave do campo para aplicar os atalhos
    st.session_state.setdefault("periodo_versao", 0)
    st.session_state.setdefault("periodo_default", (data_min, data_max))

    periodo = st.sidebar.date_input(
        "Período",
        value=st.session_state["periodo_default"],
        min_value=data_min,
        max_value=data_max,
        key=f"filtro_periodo_v{st.session_state['periodo_versao']}",
    )

    def _aplicar_atalho(dias: int | None, limite_ini, limite_fim):
        novo_ini = limite_ini if dias is None else max(limite_ini, limite_fim - timedelta(days=dias - 1))
        st.session_state["periodo_default"] = (novo_ini, limite_fim)
        st.session_state["periodo_versao"] += 1

    a1, a2, a3 = st.sidebar.columns(3)
    a1.button("7d", use_container_width=True, on_click=_aplicar_atalho, args=(7, data_min, data_max))
    a2.button("30d", use_container_width=True, on_click=_aplicar_atalho, args=(30, data_min, data_max))
    a3.button("Tudo", use_container_width=True, on_click=_aplicar_atalho, args=(None, data_min, data_max))

    if isinstance(periodo, tuple) and len(periodo) == 2:
        data_ini, data_fim = periodo
    else:
        st.sidebar.caption("Selecione também a data final.")
        data_ini, data_fim = data_min, data_max
else:
    st.sidebar.info("Ainda não há vendas. Rode o pipeline abaixo.")
    data_ini = data_fim = data_min = data_max = None

# --- Pratos
todos_pratos = sorted(vendas_full["prato"].unique().tolist()) if not vendas_full.empty else []
pratos_sel = st.sidebar.multiselect(
    "Pratos", todos_pratos, default=todos_pratos, key="filtro_pratos",
    format_func=nome_bonito,
)

# --- Insumos
insumos_full = carregar_insumos()
todos_insumos = (
    sorted(insumos_full["nome"].unique().tolist())
    if not insumos_full.empty and "nome" in insumos_full.columns else []
)
insumos_sel = st.sidebar.multiselect(
    "Insumos", todos_insumos, default=todos_insumos, key="filtro_insumos",
    format_func=nome_bonito,
)

# --- Previsão: data de início e horizonte (só recalcula ao clicar em "Aplicar")
st.sidebar.divider()
st.sidebar.subheader("Previsão")
st.session_state.setdefault("horizonte_atual", HORIZONTE_PREVISAO)
st.session_state.setdefault("inicio_previsao", date.today())

# precisa de ~30 dias de histórico antes da data de início
inicio_min = data_min + timedelta(days=31) if data_min else None
inicio_max = date.today() + timedelta(days=365)
inicio_val = st.session_state["inicio_previsao"]
if inicio_min:
    inicio_val = max(inicio_val, inicio_min)
inicio_val = min(inicio_val, inicio_max)

with st.sidebar.form("form_horizonte"):
    inicio_novo = st.date_input(
        "Início da previsão", value=inicio_val,
        min_value=inicio_min, max_value=inicio_max, format="DD/MM/YYYY",
        help="Padrão: hoje. O modelo usa só as vendas anteriores a esta data.",
    )
    H_novo = st.slider("Dias à frente", min_value=2, max_value=30,
                       value=st.session_state["horizonte_atual"], step=1)
    aplicar_h = st.form_submit_button("Aplicar", use_container_width=True)

if aplicar_h and (
    H_novo != st.session_state["horizonte_atual"]
    or inicio_novo != st.session_state["inicio_previsao"]
):
    with st.spinner("Recalculando a previsão..."):
        ok_h = _reprocessar(horizonte=H_novo, inicio=inicio_novo)
    if ok_h:
        st.session_state["horizonte_atual"] = H_novo
        st.session_state["inicio_previsao"] = inicio_novo
        _flash("success", f"Previsão recalculada: {H_novo} dias a partir de {inicio_novo:%d/%m/%Y}.")
        st.rerun()

H = st.session_state["horizonte_atual"]
INICIO_PREV = st.session_state["inicio_previsao"]
FIM_PREV = INICIO_PREV + timedelta(days=H - 1)

# --- Dados e modelo
st.sidebar.divider()
st.sidebar.subheader("Dados e modelo")

if st.sidebar.button(
    "🔄 Rodar pipeline completo", use_container_width=True,
    help="Gera os dados, treina o modelo e recalcula os alertas.",
):
    _executar_pipeline(
        ["src.data_generator", "src.train", "src.business"],
        "Pipeline",
    )

with st.sidebar.expander("⚙️ Avançado"):
    st.caption(
        "Igual ao pipeline completo, mas também refaz o estoque (`src.inventory`). "
        "Pode sobrescrever dados atuais."
    )
    confirma_regerar = st.checkbox("Confirmo que quero regerar tudo", key="confirma_regerar")
    if st.button("♻️ Regerar tudo", disabled=not confirma_regerar, use_container_width=True):
        _executar_pipeline(
            ["src.data_generator", "src.train", "src.inventory", "src.business"],
            "Regeneração completa",
        )


# ------------------------------------------------------------------
# TOPO
# ------------------------------------------------------------------
st.title("🍽️ Previsão de Demanda e Redução de Desperdício")
st.caption("Projeto TG — Machine Learning aplicado ao estoque de restaurante")

vendas = load_vendas()
alertas = load_csv("alertas.csv")
comparacao = load_csv("comparacao_estoque.csv")
necessidade_diaria = load_csv("necessidade_insumos_diaria.csv")

if vendas.empty or alertas.empty:
    st.warning("Ainda não há dados. Clique em **🔄 Rodar pipeline completo** na barra lateral.")
    st.stop()


# ------------------------------------------------------------------
# KPIs (só o que ajuda a decidir; métricas do modelo ficam em "Previsão")
# ------------------------------------------------------------------
pacote = load_modelo()
wape_final = None
if pacote and "tabela_metricas" in pacote:
    tab = pacote["tabela_metricas"]
    if "ensemble_media_ponderada" in tab.index:
        wape_final = tab.loc["ensemble_media_ponderada", "WAPE (%)"]
    else:
        wape_final = tab["WAPE (%)"].min()

alertas_f = filtrar_por_insumo(alertas, insumos_sel)
n_falta = int((alertas_f["status"] == "FALTA").sum())
n_excesso = int((alertas_f["status"] == "EXCESSO").sum())
capital_risco = float(alertas_f["custo_excesso"].sum())
custo_compra = float(
    (alertas_f.loc[alertas_f["status"] == "FALTA", "saldo"].abs()
     * alertas_f.loc[alertas_f["status"] == "FALTA", "custo_unitario"]).sum()
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Insumos em falta", n_falta,
          help="Estoque atual menor que a necessidade prevista no horizonte.")
k2.metric("Insumos com excesso", n_excesso,
          help="Estoque acima do necessário: risco de desperdício.")
k3.metric("Compra sugerida", fmt_brl(custo_compra),
          help="Custo para cobrir o que está em falta.")
k4.metric("Capital em risco", fmt_brl(capital_risco),
          help="Custo do estoque em excesso.")

st.caption(
    f"Filtros ativos — período: {data_ini} a {data_fim} | "
    f"{len(pratos_sel)}/{len(todos_pratos)} pratos | "
    f"{len(insumos_sel)}/{len(todos_insumos)} insumos"
)
st.caption(
    f"🔮 Previsão: de **{INICIO_PREV:%d/%m/%Y}** a **{FIM_PREV:%d/%m/%Y}** ({H} dias) · "
    f"última venda registrada: {vendas['data'].max():%d/%m/%Y}"
)


# ------------------------------------------------------------------
# ABAS (8 no total; o corpo de cada seção está logo abaixo)
# ------------------------------------------------------------------
tab_resumo, tab_reg, tab_prev, tab_vendas, tab_estoque, tab_desp, tab_cad, tab_chat = st.tabs([
    "🚨 Resumo",
    "✍️ Registrar",
    "🔮 Previsão",
    "🧾 Vendas",
    "📦 Estoque",
    "🗑️ Desperdício",
    "📋 Cadastros",
    "🤖 Assistente",
])

# Cada seção antiga escreve no seu bloco; seções relacionadas dividem a mesma aba.
aba_alertas = tab_resumo
aba_registrar = tab_reg
aba_prev = tab_prev.container()
aba_insumos = tab_prev.container()
aba_modelo = tab_prev.expander("🤖 Avaliação do modelo (técnico)")
aba_vendas = tab_vendas.container()   # vendas registradas (ação) primeiro...
tab_vendas.divider()
aba_hist = tab_vendas.container()     # ...gráficos do histórico depois
aba_estoque = tab_estoque
aba_desp = tab_desp
aba_cad = tab_cad
aba_chat = tab_chat

if wape_final is not None:
    with aba_modelo:
        st.metric("WAPE do modelo final (ensemble)", fmt_pct(wape_final, 2))


# ---- Registrar (vendas e movimentações de estoque)
# Dados compartilhados pelos formulários e pelas abas Estoque/Vendas
insumos_df = carregar_insumos()
lista_insumos = insumos_df["nome"].tolist()
unidade_de = dict(zip(insumos_df["nome"], insumos_df["unidade"]))
custo_de = dict(zip(insumos_df["nome"], insumos_df["custo_unitario"]))
saldos = saldos_atuais()
movs = todas_movimentacoes()


def _ultimos_movimentos(tipo: str, n: int = 5):
    """Mostra os últimos registros de um tipo de movimentação (feedback imediato)."""
    st.markdown("**Últimos registros**")
    if movs.empty or "tipo" not in movs.columns:
        st.caption("Nenhum registro ainda.")
        return
    m = movs[movs["tipo"].astype(str).str.upper().str.contains(tipo)]
    if m.empty:
        st.caption("Nenhum registro ainda.")
        return
    mostrar_tabela(m.sort_values("data", ascending=False).head(n))


with aba_registrar:
    st.subheader("Registrar")
    st.caption("Escolha o que registrar. Cada registro atualiza o estoque e os alertas.")

    sub_venda, sub_entrada, sub_perda, sub_ajuste = st.tabs([
        "🧾 Venda",
        "➕ Entrada de estoque",
        "🗑️ Perda / desperdício",
        "🧮 Contagem de estoque",
    ])

    # --------------------------------------------------------------
    # VENDA
    # --------------------------------------------------------------
    with sub_venda:
        st.caption(
            "Cada venda dá baixa automática no estoque via ficha técnica. "
            "Vendas no mesmo dia e prato são somadas."
        )

        if "msg_venda" in st.session_state:
            tipo, texto = st.session_state.pop("msg_venda")
            (st.success if tipo == "ok" else st.error)(texto)

        pra = listar_pratos()

        if pra.empty:
            st.warning("Nenhum prato cadastrado. Vá em **📋 Cadastros → Pratos**.")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            prato_v = c1.selectbox("Prato", pra["nome"].tolist(), index=None,
                                    placeholder="Selecione...", key="venda_prato_sel",
                                    format_func=nome_bonito)
            qtd_v = c2.number_input("Quantidade", min_value=0, step=1,
                                     value=None, key="venda_qtd_sel")
            data_v = c3.date_input("Data", value=date.today(), key="venda_data_sel")

            baixar = st.checkbox(
                "Baixar consumo do estoque automaticamente",
                value=True,
                key="venda_baixar",
                help="Se desmarcado, apenas registra a venda sem mexer no estoque.",
            )

            # Prévia dinâmica do consumo
            if prato_v and qtd_v and qtd_v > 0:
                ficha = ficha_do_prato(prato_v)
                if ficha.empty:
                    st.info(f"'{nome_bonito(prato_v)}' não tem ficha técnica cadastrada. "
                            "A venda será registrada sem baixa de estoque.")
                else:
                    st.markdown("**Consumo previsto:**")
                    preview = ficha.copy()
                    preview["quantidade"] = preview["quantidade_por_prato"] * qtd_v
                    preview["estoque_atual"] = preview["insumo"].apply(saldo_atual)
                    preview["saldo_pos"] = preview["estoque_atual"] - preview["quantidade"]
                    preview["ok"] = preview["saldo_pos"].apply(
                        lambda x: "✅" if x >= 0 else "⚠️"
                    )
                    mostrar_tabela(
                        preview[["insumo", "quantidade", "estoque_atual", "saldo_pos", "ok"]],
                        casas={"quantidade": 3, "estoque_atual": 3, "saldo_pos": 3},
                        titulos={"quantidade": "Consumo"},
                    )
                    st.caption(
                        "⚠️ indica que o saldo ficará negativo. "
                        "A venda ainda será registrada, mas o consumo não será baixado."
                    )

            if st.button("Registrar venda", type="primary", key="btn_registrar_venda"):
                erros = []
                if not prato_v: erros.append("Selecione o prato.")
                if qtd_v is None or qtd_v <= 0: erros.append("Quantidade deve ser > 0.")

                # Validação de estoque antes de gravar
                if not erros and baixar:
                    ficha = ficha_do_prato(prato_v)
                    faltando = []
                    for _, item in ficha.iterrows():
                        necessario = item["quantidade_por_prato"] * qtd_v
                        disponivel = saldo_atual(item["insumo"])
                        if necessario > disponivel:
                            faltando.append(
                                f"{nome_bonito(item['insumo'])}: necessário {fmt_num(necessario)}, "
                                f"disponível {fmt_num(disponivel)}"
                            )
                    if faltando:
                        erros.append(
                            "Estoque insuficiente. Ajuste a quantidade, registre uma "
                            "entrada ou contagem na aba ✍️ Registrar, ou desmarque a baixa automática."
                        )
                        erros.extend(faltando)

                if erros:
                    st.session_state["msg_venda"] = ("erro", " | ".join(erros))
                    st.rerun()
                else:
                    try:
                        registrar_venda(prato_v, int(qtd_v), data_v)
                        if baixar:
                            registrar_saida_por_venda(prato_v, int(qtd_v), data_v)
                        st.cache_data.clear()
                        st.session_state["msg_venda"] = (
                            "ok",
                            f"{int(qtd_v)}× {nome_bonito(prato_v)} registrado em {data_v:%d/%m/%Y}."
                        )
                        st.rerun()
                    except Exception as e:
                        st.session_state["msg_venda"] = ("erro", str(e))
                        st.rerun()

        st.divider()
        _ult_v = listar_vendas()
        st.markdown("**Últimas vendas registradas**")
        if _ult_v.empty:
            st.caption("Nenhuma venda registrada ainda.")
        else:
            mostrar_tabela(_ult_v.sort_values("data", ascending=False).head(5))

    # --------------------------------------------------------------
    # ENTRADA
    # --------------------------------------------------------------
    with sub_entrada:
        st.caption("Use quando chegar mercadoria: a quantidade é somada ao saldo do insumo.")

        # fora do formulário, para a unidade e o saldo atualizarem na hora
        insumo_e = st.selectbox("Insumo", lista_insumos, index=None,
                                placeholder="Selecione...", key="ent_insumo", format_func=nome_bonito)
        unidade_e = unidade_de.get(insumo_e, "un") if insumo_e else "un"
        if insumo_e:
            st.caption(
                f"Saldo atual: **{fmt_qtd(saldos.get(insumo_e, 0.0), unidade_e)}** · "
                f"custo cadastrado: {fmt_brl(custo_de.get(insumo_e, 0.0))}"
            )

        with st.form("form_entrada", clear_on_submit=True):
            c1, c2 = st.columns(2)
            qtd_e = c1.number_input(f"Quantidade ({unidade_e})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="ent_qtd")
            custo_e = c2.number_input("Custo unitário (R$)",
                                       min_value=0.0, step=0.5,
                                       value=None, key="ent_custo")

            c3, c4 = st.columns(2)
            origem_e = c3.text_input("Fornecedor / origem",
                                      value="", key="ent_origem")
            obs_e = c4.text_input("Observação", value="", key="ent_obs")

            submitted = st.form_submit_button("Registrar entrada", type="primary")

        if submitted:
            erros = []
            if not insumo_e: erros.append("Selecione um insumo.")
            if qtd_e is None or qtd_e <= 0: erros.append("Quantidade deve ser > 0.")
            if custo_e is None or custo_e < 0: erros.append("Custo inválido.")
            if erros:
                for e in erros: st.error(e)
            else:
                dialog_movimentacao(
                    "ENTRADA",
                    {
                        "Insumo": nome_bonito(insumo_e),
                        "Quantidade": fmt_qtd(qtd_e, unidade_e),
                        "Custo unitário": fmt_brl(custo_e),
                        "Origem": origem_e or "—",
                        "Observação": obs_e or "—",
                    },
                    lambda: registrar_entrada(
                        insumo_e, qtd_e, custo_e,
                        origem_e or "fornecedor", obs_e,
                    ),
                )

        st.divider()
        _ultimos_movimentos("ENTRADA")

    # --------------------------------------------------------------
    # PERDA
    # --------------------------------------------------------------
    with sub_perda:
        st.caption("Use para descarte (validade, preparo, sobra, avaria): "
                   "subtrai do saldo e entra na análise de desperdício.")

        insumo_p = st.selectbox("Insumo", lista_insumos, index=None,
                                placeholder="Selecione...", key="per_insumo", format_func=nome_bonito)
        unidade_p = unidade_de.get(insumo_p, "un") if insumo_p else "un"
        if insumo_p:
            st.caption(f"Saldo atual: **{fmt_qtd(saldos.get(insumo_p, 0.0), unidade_p)}**")

        with st.form("form_perda", clear_on_submit=True):
            c1, c2 = st.columns(2)
            qtd_p = c1.number_input(f"Quantidade ({unidade_p})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="per_qtd")
            motivo_p = c2.selectbox("Motivo",
                                     ["validade", "preparo", "sobra", "avaria", "outro"],
                                     index=None, placeholder="Selecione...",
                                     key="per_motivo", format_func=rotulo)

            obs_p = st.text_input("Observação", value="", key="per_obs")
            submitted = st.form_submit_button("Registrar perda", type="primary")

        if submitted:
            erros = []
            if not insumo_p: erros.append("Selecione um insumo.")
            if qtd_p is None or qtd_p <= 0: erros.append("Quantidade deve ser > 0.")
            if not motivo_p: erros.append("Selecione um motivo.")
            if erros:
                for e in erros: st.error(e)
            else:
                dialog_movimentacao(
                    "PERDA",
                    {
                        "Insumo": nome_bonito(insumo_p),
                        "Quantidade": fmt_qtd(qtd_p, unidade_p),
                        "Motivo": rotulo(motivo_p),
                        "Observação": obs_p or "—",
                    },
                    lambda: registrar_perda(insumo_p, qtd_p, motivo_p, obs_p),
                )

        st.divider()
        _ultimos_movimentos("PERDA")

    # --------------------------------------------------------------
    # CONTAGEM / AJUSTE
    # --------------------------------------------------------------
    with sub_ajuste:
        st.caption("Use depois de contar o estoque físico: o saldo do sistema "
                   "é corrigido para o valor contado.")

        c1, c2 = st.columns(2)
        insumo_a = c1.selectbox("Insumo", lista_insumos, index=None,
                                placeholder="Selecione...", key="aj_insumo", format_func=nome_bonito)
        unidade_a = unidade_de.get(insumo_a, "un") if insumo_a else "un"
        saldo_a = saldos.get(insumo_a, 0.0) if insumo_a else 0.0
        c2.metric("Saldo no sistema", fmt_qtd(saldo_a, unidade_a) if insumo_a else "—")

        with st.form("form_ajuste", clear_on_submit=True):
            qtd_a = st.number_input(f"Quantidade contada ({unidade_a})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="aj_qtd")
            obs_a = st.text_input("Observação", value="", key="aj_obs")
            submitted = st.form_submit_button("Registrar contagem", type="primary")

        if submitted:
            erros = []
            if not insumo_a: erros.append("Selecione um insumo.")
            if qtd_a is None: erros.append("Informe a quantidade contada.")
            if erros:
                for e in erros: st.error(e)
            else:
                delta = qtd_a - saldo_a
                dialog_movimentacao(
                    "AJUSTE",
                    {
                        "Insumo": nome_bonito(insumo_a),
                        "Saldo no sistema": fmt_qtd(saldo_a, unidade_a),
                        "Contagem física": fmt_qtd(qtd_a, unidade_a),
                        "Diferença": ("+" if delta > 0 else "") + fmt_qtd(delta, unidade_a),
                        "Observação": obs_a or "—",
                    },
                    lambda: registrar_ajuste(insumo_a, qtd_a, obs_a),
                )

        st.divider()
        _ultimos_movimentos("AJUSTE")


# ---- Histórico
with aba_hist:
    st.subheader("Vendas históricas")
    v = filtrar_vendas(vendas, data_ini, data_fim, pratos_sel)

    if v.empty:
        st.warning("Nenhum dado no filtro atual.")
    else:
        # ---------- Controle ----------
        agregacao = st.radio(
            "Agregação",
            ["Diária", "Semanal", "Mensal"],
            index=1,
            horizontal=True,
            key="hist_agreg",
        )

        freq_map = {"Diária": "D", "Semanal": "W-MON", "Mensal": "MS"}
        fmt_eixo = {"Diária": "%d/%m", "Semanal": "S%W/%y", "Mensal": "%m/%Y"}[agregacao]

        # ---------- Agregação total e por prato ----------
        base = v.set_index("data")

        total = (
            base["quantidade"]
            .resample(freq_map[agregacao])
            .sum()
            .reset_index()
            .rename(columns={"quantidade": "total"})
        )

        por_prato = (
            base.groupby("prato")["quantidade"]
            .resample(freq_map[agregacao])
            .sum()
            .reset_index()
        )

        # Remove buckets parciais no início e fim
        if agregacao in ("Semanal", "Mensal"):
            ini = total["data"].min()
            fim = total["data"].max()
            total = total[(total["data"] > ini) & (total["data"] < fim)]
            por_prato = por_prato[
                (por_prato["data"] > ini) & (por_prato["data"] < fim)
            ]

        # ---------- Gráfico 1: total ----------
        st.markdown("**Total de vendas no período**")
        fig_total = px.line(total, x="data", y="total", labels=ROTULOS_GRAFICO)
        fig_total.update_traces(
            line=dict(width=2.5, color="#4a90d9"),
            name="Total",
        )

        # Linha suavizada
        if len(total) > 7:
            total["suave"] = total["total"].rolling(7, min_periods=1).mean()
            fig_total.add_scatter(
                x=total["data"], y=total["suave"],
                mode="lines", name="Tendência (média 7)",
                line=dict(dash="dash", width=1.5, color="#f0a020"),
            )

        fig_total.update_xaxes(tickformat=fmt_eixo, nticks=8)
        fig_total.update_yaxes(rangemode="tozero", title="Pratos vendidos")
        fig_total.update_layout(
            height=350,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        grafico(fig_total)

        # ---------- Gráfico 2: composição ----------
        st.markdown("**Composição por prato**")
        fig_area = px.area(
            bonito(por_prato), x="data", y="quantidade", color="prato",
            labels=ROTULOS_GRAFICO,
        )
        fig_area.update_xaxes(tickformat=fmt_eixo, nticks=8)
        fig_area.update_yaxes(rangemode="tozero", title="Pratos vendidos")
        fig_area.update_layout(
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        grafico(fig_area)

        # ---------- Vendas médias por dia da semana ----------
        st.subheader("Vendas médias por dia da semana")
        v2 = v.copy()
        v2["dia_semana"] = v2["data"].dt.weekday
        trad = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
        v2["dia_semana_pt"] = v2["dia_semana"].map(trad)
        ordem_pt = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

        media_dow = (
            v2.groupby(["dia_semana_pt", "prato"], observed=True)["quantidade"]
              .mean()
              .reset_index()
        )
        fig2 = px.bar(
            bonito(media_dow), x="dia_semana_pt", y="quantidade",
            color="prato", barmode="group",
            category_orders={"dia_semana_pt": ordem_pt},
            labels=ROTULOS_GRAFICO,
        )
        fig2.update_xaxes(title="")
        fig2.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        grafico(fig2)

        # ---------- Feriado vs. dia comum ----------
        st.subheader("Comparação feriado vs. dia comum")
        v2["tipo_dia"] = v2["feriado"].map({0: "Dia comum", 1: "Feriado"})
        media_fer = (
            v2.groupby(["tipo_dia", "prato"], observed=True)["quantidade"]
              .mean()
              .reset_index()
        )
        fig3 = px.bar(
            bonito(media_fer), x="tipo_dia", y="quantidade",
            color="prato", barmode="group", labels=ROTULOS_GRAFICO,
        )
        fig3.update_xaxes(title="")
        fig3.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        grafico(fig3)

        with st.expander("Ver tabela bruta"):
            mostrar_tabela(v.sort_values("data", ascending=False))

with aba_vendas:
    st.subheader("Vendas registradas")
    st.caption("Para registrar uma venda nova, use a aba **✍️ Registrar**.")

    vendas_df = listar_vendas()
    if vendas_df.empty:
        st.info("Nenhuma venda registrada ainda.")
    else:
        vendas_df["data"] = pd.to_datetime(vendas_df["data"]).dt.date
        hoje = date.today()
        dmin = vendas_df["data"].min()
        dmax = max(vendas_df["data"].max(), hoje)

        # Inicializa o par de datas em session_state
        if "venda_periodo" not in st.session_state:
            st.session_state["venda_periodo"] = (dmin, dmax)
        if "venda_versao" not in st.session_state:
            st.session_state["venda_versao"] = 0

        def _atalho_venda(ini, fim):
            st.session_state["venda_periodo"] = (ini, fim)
            st.session_state["venda_versao"] += 1

        # Botões de atalho
        c1, c2, c3, c4 = st.columns(4)
        c1.button("Hoje", use_container_width=True,
                  on_click=_atalho_venda, args=(hoje, hoje))
        c2.button("7 dias", use_container_width=True,
                  on_click=_atalho_venda,
                  args=(max(dmin, hoje - timedelta(days=6)), hoje))
        c3.button("30 dias", use_container_width=True,
                  on_click=_atalho_venda,
                  args=(max(dmin, hoje - timedelta(days=29)), hoje))
        c4.button("Tudo", use_container_width=True,
                  on_click=_atalho_venda, args=(dmin, dmax))

        # Date inputs (key muda com a versão)
        c1, c2, c3 = st.columns([2, 2, 2])
        f_ini = c1.date_input(
            "De",
            value=st.session_state["venda_periodo"][0],
            min_value=dmin,
            max_value=dmax,
            key=f"ven_ini_v{st.session_state['venda_versao']}",
        )
        f_fim = c2.date_input(
            "Até",
            value=st.session_state["venda_periodo"][1],
            min_value=dmin,
            max_value=dmax,
            key=f"ven_fim_v{st.session_state['venda_versao']}",
        )
        f_prato = c3.multiselect(
            "Prato",
            sorted(vendas_df["prato"].unique().tolist()),
            default=sorted(vendas_df["prato"].unique().tolist()),
            format_func=nome_bonito,
        )

        filtro = vendas_df[
            (vendas_df["data"] >= f_ini)
            & (vendas_df["data"] <= f_fim)
            & (vendas_df["prato"].isin(f_prato))
        ].sort_values("data", ascending=False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Registros", fmt_int(len(filtro)))
        m2.metric("Total de pratos vendidos", fmt_int(filtro["quantidade"].sum()))
        m3.metric("Média por registro",
                  fmt_num(filtro['quantidade'].mean(), 1) if len(filtro) else "—")

        mostrar_tabela(filtro.head(500))

        st.divider()
        with st.expander("🗑️ Remover venda"):
            c1, c2 = st.columns(2)
            pra_r = c1.selectbox("Prato",
                                  sorted(vendas_df["prato"].unique().tolist()),
                                  key="rm_venda_prato", format_func=nome_bonito)
            datas_disp = sorted(vendas_df[vendas_df["prato"] == pra_r]["data"].unique())
            data_r = c2.selectbox("Data", datas_disp, key="rm_venda_data",
                                  format_func=lambda d: d.strftime("%d/%m/%Y"))
            st.caption(
                "⚠️ Remover a venda **não** estorna o consumo do estoque. "
                "Se quiser estornar, use **✍️ Registrar → Contagem de estoque**."
            )
            if st.button("Confirmar remoção", key="btn_rm_venda"):
                remover_venda(pra_r, data_r)
                st.success("Venda removida.")
                st.cache_data.clear()
                st.rerun()

# ---- Modelo
with aba_modelo:
    st.subheader("Comparação de modelos")
    if pacote and "tabela_metricas" in pacote:
        tm = pacote["tabela_metricas"].copy()
        tm.index = [NOMES_MODELOS.get(i, nome_bonito(i)) for i in tm.index]
        tm.index.name = "Modelo"
        st.dataframe(
            tm.style.highlight_min(
                axis=0, subset=["MAE", "RMSE", "WAPE (%)"], color="#87A96B"
            ).format(_formatar_fmt_qualquer),
            use_container_width=True,
        )
        melhor = pacote.get("melhor_modelo", "—")
        wape_melhor = pacote["tabela_metricas"].loc[melhor, "WAPE (%)"] if melhor in pacote["tabela_metricas"].index else None
        msg = f"Modelo final: **{NOMES_MODELOS.get(melhor, nome_bonito(melhor))}**"
        if wape_melhor is not None:
            msg += f" — WAPE {fmt_pct(wape_melhor, 2)}"
        st.success(msg)
    else:
        st.info("Treine o modelo para visualizar as métricas.")


# ---- Previsão
with aba_prev:
    from src.predict import prever_proximos_dias

    H = st.session_state["horizonte_atual"]

    st.subheader(f"Previsão dos próximos {H} dias")
    st.caption(
        f"Previsão de **{INICIO_PREV:%d/%m/%Y}** a **{FIM_PREV:%d/%m/%Y}**. "
        "Use **Previsão** na barra lateral para alterar o início e o número de dias."
    )

    _ultima_venda = vendas["data"].max().date()
    _lacuna = (INICIO_PREV - _ultima_venda).days - 1
    if _lacuna > 0:
        st.warning(
            f"A última venda registrada é de {_ultima_venda:%d/%m/%Y}. Os {_lacuna} dia(s) "
            f"até {INICIO_PREV:%d/%m/%Y} foram estimados pelo próprio modelo, então a "
            f"incerteza é maior{' (lacuna grande: trate como tendência, não como número exato)' if _lacuna > 30 else ''}. "
            "Registrar as vendas mais recentes melhora a previsão."
        )
    elif INICIO_PREV <= _ultima_venda:
        st.info(
            "O início está dentro do período com vendas reais: a previsão usa só o "
            "histórico anterior a essa data, o que permite comparar com o que de fato vendeu."
        )

    c1, c2 = st.columns([3, 1])
    with c2:
        tipo_vis = st.radio("Visualização", ["Linha", "Área", "Barras"],
                             index=0, key="prev_vis")

    try:
        prev = prever_proximos_dias(H, INICIO_PREV)
        prev_f = prev[prev["prato"].isin(pratos_sel)]

        if prev.empty:
            st.warning("Histórico insuficiente antes da data de início escolhida "
                       "(são necessários cerca de 30 dias de vendas).")
        elif prev_f.empty:
            st.warning("Nenhum prato no filtro atual.")
        else:
            fig_kwargs = dict(
                x="data", y="previsao", color="prato",
                title=f"Previsão — próximos {H} dias",
                labels=ROTULOS_GRAFICO,
            )
            if tipo_vis == "Linha":
                fig3 = px.line(bonito(prev_f), markers=True, **fig_kwargs)
            elif tipo_vis == "Área":
                fig3 = px.area(bonito(prev_f), **fig_kwargs)
            else:
                fig3 = px.bar(bonito(prev_f), barmode="group", **fig_kwargs)
            grafico(fig3)

            # Histórico + previsão
            v = filtrar_vendas(vendas, data_ini, data_fim, pratos_sel)
            ult_30 = v[v["data"] >= (v["data"].max() - pd.Timedelta(days=29))]
            hist_agg = ult_30.groupby(["data", "prato"], observed=True)["quantidade"] \
                             .sum().reset_index()
            hist_agg["tipo"] = "Histórico"
            prev_f2 = prev_f.rename(columns={"previsao": "quantidade"}).copy()
            prev_f2["tipo"] = "Previsão"
            combo = pd.concat(
                [hist_agg[["data", "prato", "quantidade", "tipo"]],
                 prev_f2[["data", "prato", "quantidade", "tipo"]]],
                ignore_index=True,
            )
            fig4 = px.line(
                bonito(combo), x="data", y="quantidade", color="prato",
                line_dash="tipo",
                title="Últimos 30 dias + previsão",
                labels=ROTULOS_GRAFICO,
            )
            grafico(fig4)

            st.markdown(f"**Tabela de previsão — {H} dias**")
            mostrar_tabela(
                prev_f.pivot(index="data", columns="prato", values="previsao"),
                indice=True, pivot=True, formato=fmt_int,
            )

            # Totais por prato no horizonte
            st.markdown("**Total previsto por prato**")
            totais = (
                prev_f.groupby("prato", observed=True)["previsao"]
                      .agg(["sum", "mean", "min", "max"])
                      .round(1)
                      .rename(columns={
                          "sum": "total",
                          "mean": "média/dia",
                          "min": "mínimo",
                          "max": "máximo",
                      })
            )
            mostrar_tabela(
                totais, indice=True,
                casas={"total": 0, "média/dia": 1, "mínimo": 0, "máximo": 0},
            )

    except Exception as e:
        st.error(f"Não foi possível gerar previsão: {e}")


# ---- Insumos
with aba_insumos:
    H = st.session_state["horizonte_atual"]
    st.subheader(f"Necessidade diária de insumos — próximos {H} dias")

    nd = filtrar_por_insumo(necessidade_diaria, insumos_sel)
    if nd.empty:
        st.warning("Nenhum insumo no filtro atual.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            metrica = st.radio("Métrica", ["Quantidade", "Custo"],
                                index=0, key="ins_metrica")
            ordem = st.radio("Ordenar por", ["Insumo", "Total"],
                              index=0, key="ins_ordem")

        insumos_df = carregar_insumos()
        nd = nd.merge(insumos_df[["nome", "custo_unitario"]],
                      left_on="insumo", right_on="nome", how="left")
        nd["custo"] = nd["quantidade_necessaria"] * nd["custo_unitario"]
        col_y = "quantidade_necessaria" if metrica == "Quantidade" else "custo"
        ylabel = "kg" if metrica == "Quantidade" else "R$"

        ordem_insumos = (
            nd.groupby("insumo", observed=True)[col_y].sum()
              .sort_values(ascending=False).index.tolist()
        )
        if ordem == "Insumo":
            ordem_insumos = sorted(ordem_insumos)

        fig4 = px.bar(
            bonito(nd), x="data", y=col_y, color="insumo", barmode="stack",
            category_orders={"insumo": [nome_bonito(i) for i in ordem_insumos]},
            title=f"Consumo previsto por dia — {metrica}",
            labels=ROTULOS_GRAFICO,
        )
        fig4.update_yaxes(title=ylabel)
        grafico(fig4)

        st.markdown("**Tabela de necessidade diária**")
        pivot = nd.pivot_table(
            index="data", columns="insumo",
            values=col_y, aggfunc="sum",
        ).round(2)
        mostrar_tabela(pivot, indice=True, pivot=True,
                       formato=fmt_brl if metrica == "Custo" else fmt_num)

        st.markdown("**Total por insumo no período**")
        total = (
            nd.groupby("insumo", observed=True)[col_y].sum()
              .sort_values(ascending=False).round(2).reset_index()
        )
        mostrar_tabela(total)


# ---- Alertas
with aba_alertas:
    H = st.session_state["horizonte_atual"]
    st.subheader(f"Status dos insumos — próximos {H} dias")
    st.caption(
        f"A **necessidade** de cada insumo é calculada a partir da previsão "
        f"de demanda dos próximos {H} dias, convertida via ficha técnica. "
        f"O **saldo** é a diferença entre o estoque atual e essa necessidade."
    )

    al = filtrar_por_insumo(alertas, insumos_sel)

    c1, c2 = st.columns([1, 1])
    with c2:
        status_sel = st.multiselect(
            "Status",
            ["FALTA", "OK", "EXCESSO"],
            default=["FALTA", "OK", "EXCESSO"],
            key="al_status",
            format_func=rotulo,
        )
        ordenar = st.radio("Ordenar por", ["Status", "Saldo", "Custo"],
                            index=0, key="al_ord")

    al = al[al["status"].isin(status_sel)]
    if ordenar == "Saldo":
        al = al.sort_values("saldo")
    elif ordenar == "Custo":
        al = al.sort_values("custo_excesso", ascending=False)

    def cor_status(s):
        return {
            "Falta": "background-color: #E53E3E",
            "Excesso": "background-color: #FFB300",
            "OK": "background-color: #87A96B",
        }.get(s, "")

    cols_show = ["insumo", "unidade", "estoque_atual", "necessidade_total",
                 "saldo", "status", "custo_excesso"]
    mostrar_tabela(
        al[cols_show],
        titulos={
            "necessidade_total": f"Necessidade ({H} dias)",
            "saldo": f"Saldo ({H} dias)",
        },
        destaque=(cor_status, ["status"]),
    )

    st.divider()
    st.subheader("Sugestão de compra")
    falta = al[al["status"] == "FALTA"].copy()
    if falta.empty:
        st.success("Nenhuma compra necessária no filtro atual.")
    else:
        falta["qtd_sugerida"] = (-falta["saldo"] * 1.10).round(2)
        falta["custo_estimado"] = (falta["qtd_sugerida"] * falta["custo_unitario"]).round(2)
        mostrar_tabela(
            falta[["insumo", "unidade", "qtd_sugerida", "custo_unitario", "custo_estimado"]],
            titulos={"qtd_sugerida": "Qtd. sugerida (+10%)"},
        )
        st.info(f"Total estimado: **{fmt_brl(falta['custo_estimado'].sum())}**")

        fig_c = px.bar(bonito(falta), x="insumo", y="custo_estimado",
                       title="Custo de compra por insumo (R$)", labels=ROTULOS_GRAFICO)
        grafico(fig_c)

    if not al[al["status"] == "EXCESSO"].empty:
        st.divider()
        st.subheader("⚠️ Insumos com excesso (risco de desperdício)")
        exc = al[al["status"] == "EXCESSO"]
        st.warning(
            f"{len(exc)} insumo(s) com estoque acima do necessário. "
            f"Capital em risco: **{fmt_brl(exc['custo_excesso'].sum())}**."
        )
    
with aba_estoque:
    # ---------- Saldos atuais ----------
    st.subheader("Saldos atuais")
    st.caption("Para registrar entradas, perdas ou contagens, use a aba **✍️ Registrar**.")
    df_saldos = pd.DataFrame(
        [{"insumo": k, "saldo_atual": v} for k, v in saldos.items()]
    ).merge(insumos_df, left_on="insumo", right_on="nome", how="left")
    df_saldos["valor_total"] = (
        df_saldos["saldo_atual"] * df_saldos["custo_unitario"]
    ).round(2)
    mostrar_tabela(
        df_saldos[["insumo", "saldo_atual", "unidade", "custo_unitario", "valor_total"]],
    )
    st.caption(f"Valor total em estoque: **{fmt_brl(df_saldos['valor_total'].sum())}**")

    st.divider()
    st.subheader("Histórico de movimentações")

    if movs.empty:
        st.info("Nenhuma movimentação registrada ainda.")
    else:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        f_tipo = c1.multiselect(
            "Tipo", sorted(movs["tipo"].unique().tolist()),
            default=sorted(movs["tipo"].unique().tolist()),
            format_func=rotulo,
        )
        f_insumo = c2.multiselect(
            "Insumo", sorted(movs["insumo"].unique().tolist()),
            default=[i for i in sorted(movs["insumo"].unique().tolist())
                     if i in insumos_sel],
            format_func=nome_bonito,
        )
        datas_mov = pd.to_datetime(movs["data"]).dt.date
        f_ini = c3.date_input("De", value=datas_mov.min(),
                              min_value=datas_mov.min(), max_value=datas_mov.max())
        limite = c4.number_input("Máx.", min_value=10, max_value=1000,
                                  value=200, step=10)

        filtro = movs[
            movs["tipo"].isin(f_tipo)
            & movs["insumo"].isin(f_insumo)
            & (pd.to_datetime(movs["data"]).dt.date >= f_ini)
        ]
        mostrar_tabela(filtro.sort_values("data", ascending=False).head(int(limite)))
        st.caption(f"{len(filtro)} movimentações no filtro atual.")

with aba_desp:
    st.subheader("Análise de desperdício")

    kpis = taxa_desperdicio()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taxa de desperdício", fmt_pct(kpis['taxa_desperdicio_pct'], 2))
    c2.metric("Custo total de perdas", fmt_brl(kpis['custo_total_perdas']))
    c3.metric("Custo total de entradas", fmt_brl(kpis['custo_total_entradas']))
    c4.metric("Registros de perda", fmt_int(kpis["n_registros_perda"]))

    st.caption(
        "A taxa de desperdício representa o percentual do valor comprado "
        "que foi perdido. Valores acima de 10% indicam oportunidade de melhoria."
    )

    st.divider()

    df_motivo = filtrar_por_insumo(load_csv("desperdicio_por_motivo.csv"), insumos_sel)
    df_insumo = filtrar_por_insumo(load_csv("desperdicio_por_insumo.csv"), insumos_sel)
    df_dia = load_csv("desperdicio_por_dia.csv")
    df_balanco = filtrar_por_insumo(load_csv("balanco_insumos.csv"), insumos_sel)
    df_ctr = filtrar_por_insumo(load_csv("consumo_teorico_vs_real.csv"), insumos_sel)

    if df_dia.empty:
        st.info(
            "Nenhuma perda registrada ainda. Use a aba **✍️ Registrar** para "
            "registrar desperdícios e ver as análises aqui."
        )
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Perdas por motivo**")
            if not df_motivo.empty:
                metrica_d = st.radio("Métrica", ["Custo (R$)", "Quantidade"],
                                      horizontal=True, key="d_mot")
                col_v = "custo" if metrica_d == "Custo (R$)" else "quantidade"
                fig_m = px.pie(df_motivo.assign(motivo=df_motivo["motivo"].map(rotulo)),
                               names="motivo", values=col_v, labels=ROTULOS_GRAFICO,
                               hole=0.4,
                               title=f"Distribuição por motivo — {metrica_d}")
                grafico(fig_m)

        with c2:
            st.markdown("**Top insumos por perda**")
            if not df_insumo.empty:
                top_n = st.slider("Top N", 3, 15, 10, key="d_topn")
                fig_i = px.bar(
                    bonito(df_insumo.head(top_n)), x="custo", y="insumo",
                    orientation="h",
                    title=f"Top {top_n} insumos por custo de perda",
                    labels=ROTULOS_GRAFICO,
                )
                fig_i.update_layout(yaxis={"categoryorder": "total ascending"})
                grafico(fig_i)

        if not df_dia.empty:
            st.markdown("**Evolução diária das perdas**")
            df_dia["data"] = pd.to_datetime(df_dia["data"])
            df_dia_f = df_dia[
                (df_dia["data"].dt.date >= data_ini)
                & (df_dia["data"].dt.date <= data_fim)
            ]
            if not df_dia_f.empty:
                fig_d = px.line(df_dia_f, x="data", y="custo", markers=True,
                                title="Custo de perdas por dia (R$)",
                                labels=ROTULOS_GRAFICO)
                grafico(fig_d)

        st.divider()
        st.subheader("Consumo teórico vs. real")
        st.caption(
            "Teórico = vendas × ficha técnica. Real = saídas por venda. "
            "Diferença positiva indica desperdício oculto ou erro de porção."
        )
        if not df_ctr.empty:
            fig_ctr = px.bar(
                bonito(df_ctr).rename(columns={"teorico": "Consumo teórico",
                                           "saida_real": "Consumo real"}),
                x="insumo", y=["Consumo teórico", "Consumo real"],
                barmode="group",
                title="Consumo teórico vs. real (kg)",
                labels=ROTULOS_GRAFICO,
            )
            grafico(fig_ctr)
            mostrar_tabela(df_ctr)

        st.divider()
        st.subheader("Balanço por insumo")
        st.caption(
            "Verificação contábil: **comprado = consumido + perdido + saldo** "
            "(± ajustes)."
        )
        if not df_balanco.empty:
            mostrar_tabela(df_balanco)


# ---- Cadastros
UNIDADES = ["kg", "g", "L", "mL", "un"]


def _vazio(x) -> bool:
    return pd.isna(x) or str(x).strip() == ""


def _igual(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def _sincronizar(original, editado, chave, adicionar, atualizar, remover):
    """Compara a tabela original com a editada e aplica as diferenças.

    Linhas que sumiram -> remover(linha); linhas alteradas -> atualizar(antiga, nova);
    linhas novas -> adicionar(linha). Retorna (qtd_aplicada, lista_de_erros).
    """
    aplicadas, erros = 0, []

    for idx in original.index.difference(editado.index):
        linha = original.loc[idx]
        try:
            remover(linha)
            aplicadas += 1
        except Exception as e:
            erros.append(f"Remover '{linha[chave]}': {e}")

    for idx in editado.index.intersection(original.index):
        antiga, nova = original.loc[idx], editado.loc[idx]
        if all(_igual(antiga[c], nova[c]) for c in original.columns):
            continue
        if _vazio(nova[chave]):
            erros.append(f"'{antiga[chave]}': o nome não pode ficar vazio.")
            continue
        try:
            atualizar(antiga, nova)
            aplicadas += 1
        except Exception as e:
            erros.append(f"Atualizar '{antiga[chave]}': {e}")

    for idx in editado.index.difference(original.index):
        nova = editado.loc[idx]
        if _vazio(nova[chave]):
            continue  # linha em branco, ignora
        try:
            adicionar(nova)
            aplicadas += 1
        except Exception as e:
            erros.append(f"Adicionar '{nova[chave]}': {e}")

    return aplicadas, erros


def _tabela_editavel(base, nome, config, adicionar, atualizar, remover, chave="nome"):
    """Tabela editável com botão Salvar. Edite direto nas células, use a última
    linha em branco para adicionar e selecione a linha + Delete para remover."""
    base = base.reset_index(drop=True)
    vkey = f"versao_{nome}"
    versao = st.session_state.get(vkey, 0)  # muda a chave do editor após salvar

    editado = st.data_editor(
        base,
        key=f"editor_{nome}_{versao}",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config=config,
    )

    if st.button("💾 Salvar alterações", type="primary", key=f"salvar_{nome}_{versao}"):
        aplicadas, erros = _sincronizar(base, editado, chave, adicionar, atualizar, remover)
        if erros and not aplicadas:
            for e in erros:
                st.error(e)  # mantém o que a pessoa digitou
        elif aplicadas or erros:
            st.cache_data.clear()
            if aplicadas:
                _flash("success", f"{aplicadas} alteração(ões) salva(s).")
            for e in erros[:5]:
                _flash("error", e)
            st.session_state[vkey] = versao + 1
            st.rerun()
        else:
            st.info("Nenhuma alteração para salvar.")


with aba_cad:
    st.subheader("Cadastros")
    st.caption(
        "Clique numa célula para editar. Para **adicionar**, use a última linha em branco; "
        "para **remover**, selecione a linha e aperte Delete. Depois clique em **Salvar**. "
        "Alterações aqui afetam o pipeline inteiro: depois de editar, recomendo "
        "**🔄 Rodar pipeline completo** para retreinar o modelo. "
        "Os nomes nesta aba aparecem como estão salvos (ex.: `frango_grelhado`); "
        "nas outras telas eles são exibidos formatados."
    )

    sub_ins, sub_pra, sub_fic = st.tabs(
        ["🥕 Insumos", "🍽️ Pratos", "📖 Ficha técnica"]
    )

    # --------------------------------------------------------------
    # INSUMOS
    # --------------------------------------------------------------
    with sub_ins:
        st.caption("Só é possível remover insumos que não estejam em nenhuma ficha técnica. "
                   "Para trocar o nome de um insumo, crie um novo e remova o antigo.")
        cols_ins = ["nome", "unidade", "custo_unitario"]
        ins = listar_insumos().reindex(columns=cols_ins)

        def _ins_add(r):
            if _vazio(r["unidade"]):
                raise ValueError("escolha a unidade.")
            custo = 0.0 if pd.isna(r["custo_unitario"]) else float(r["custo_unitario"])
            adicionar_insumo(str(r["nome"]).strip(), r["unidade"], custo)

        def _ins_upd(antiga, nova):
            if antiga["nome"] != nova["nome"]:
                raise ValueError("não é possível renomear um insumo.")
            if _vazio(nova["unidade"]):
                raise ValueError("escolha a unidade.")
            custo = nova["custo_unitario"] if not pd.isna(nova["custo_unitario"]) else antiga["custo_unitario"]
            editar_insumo(antiga["nome"], nova["unidade"], float(custo))

        _tabela_editavel(
            ins, "insumos",
            config={
                "nome": st.column_config.TextColumn("Insumo", required=True),
                "unidade": st.column_config.SelectboxColumn("Unidade", options=UNIDADES, required=True),
                "custo_unitario": st.column_config.NumberColumn(
                    "Custo unitário (R$)", min_value=0.0, step=0.5, format="R$ %.2f"),
            },
            adicionar=_ins_add,
            atualizar=_ins_upd,
            remover=lambda r: remover_insumo(r["nome"]),
        )

    # --------------------------------------------------------------
    # PRATOS
    # --------------------------------------------------------------
    with sub_pra:
        st.caption("Só é possível remover pratos sem ficha técnica e sem vendas registradas. "
                   "Para renomear, edite o nome direto na tabela.")
        pra = listar_pratos().reindex(columns=["nome"])

        _tabela_editavel(
            pra, "pratos",
            config={"nome": st.column_config.TextColumn("Prato", required=True)},
            adicionar=lambda r: adicionar_prato(str(r["nome"]).strip()),
            atualizar=lambda a, n: editar_prato(a["nome"], str(n["nome"]).strip()),
            remover=lambda r: remover_prato(r["nome"]),
        )

    # --------------------------------------------------------------
    # FICHA TÉCNICA
    # --------------------------------------------------------------
    with sub_fic:
        pra = listar_pratos()
        ins = listar_insumos()

        if pra.empty:
            st.info("Cadastre ao menos um prato antes.")
        elif ins.empty:
            st.info("Cadastre ao menos um insumo antes.")
        else:
            prato_sel = st.selectbox("Prato", pra["nome"].tolist(), key="fic_prato",
                                     format_func=nome_bonito)
            ficha = ficha_do_prato(prato_sel)

            if ficha.empty:
                st.info(f"'{nome_bonito(prato_sel)}' ainda não tem ficha técnica. Adicione os itens abaixo.")
            else:
                merge = ficha.merge(ins, left_on="insumo", right_on="nome", how="left")
                merge["custo"] = merge["quantidade_por_prato"] * merge["custo_unitario"]
                st.metric(f"Custo de insumos por {nome_bonito(prato_sel)}", fmt_brl(merge['custo'].sum()))
                with st.expander("Ver custo por item"):
                    mostrar_tabela(
                        merge[["insumo", "quantidade_por_prato", "unidade",
                               "custo_unitario", "custo"]],
                        titulos={"custo": "Custo por prato"},
                    )

            def _fic_add(r):
                if _vazio(r["quantidade_por_prato"]):
                    raise ValueError("informe a quantidade por prato.")
                adicionar_item_ficha(prato_sel, r["insumo"], float(r["quantidade_por_prato"]))

            def _fic_upd(antiga, nova):
                if _vazio(nova["quantidade_por_prato"]):
                    raise ValueError("informe a quantidade por prato.")
                if antiga["insumo"] != nova["insumo"]:
                    remover_item_ficha(prato_sel, antiga["insumo"])
                _fic_add(nova)

            _tabela_editavel(
                ficha.reindex(columns=["insumo", "quantidade_por_prato"]),
                f"ficha_{prato_sel}",
                config={
                    "insumo": st.column_config.SelectboxColumn(
                        "Insumo", options=ins["nome"].tolist(), required=True),
                    "quantidade_por_prato": st.column_config.NumberColumn(
                        "Quantidade por prato", min_value=0.0, step=0.01, format="%.3f"),
                },
                adicionar=_fic_add,
                atualizar=_fic_upd,
                remover=lambda r: remover_item_ficha(prato_sel, r["insumo"]),
                chave="insumo",
            )


# ------------------------------------------------------------------
# ASSISTENTE IA (chat)
# ------------------------------------------------------------------
def _df_para_texto(df: pd.DataFrame, max_linhas: int = 25) -> str:
    """Converte um DataFrame pequeno em texto legível para o contexto da IA."""
    if df is None or df.empty:
        return "(vazio)"
    d = df.head(max_linhas).copy()
    for c in d.columns:
        if c in _COL_NOMES:
            d[c] = d[c].map(nome_bonito)
        elif c in _COL_ROTULOS:
            d[c] = d[c].map(rotulo)
    return d.to_string(index=False)


def _construir_contexto_ia(
    vendas_df: pd.DataFrame,
    alertas_df: pd.DataFrame,
    necessidade_df: pd.DataFrame,
    saldos_dict: dict,
    insumos_info: pd.DataFrame,
    kpis_desperdicio: dict,
    data_ini,
    data_fim,
    pratos_sel,
    insumos_sel,
    H: int,
    inicio_prev: date,
    fim_prev: date,
) -> str:
    """Monta um resumo textual com os dados de negócio (sem detalhes de ML)."""
    linhas = []
    linhas.append("=== CONTEXTO DO RESTAURANTE (dados atuais do sistema) ===")
    linhas.append(f"Data de hoje: {date.today():%d/%m/%Y}")
    linhas.append(
        f"Filtros da tela: período {data_ini} a {data_fim} | "
        f"{len(pratos_sel)} pratos | {len(insumos_sel)} insumos"
    )
    linhas.append(
        f"Horizonte de previsão: {H} dias ({inicio_prev:%d/%m/%Y} a {fim_prev:%d/%m/%Y})"
    )
    linhas.append("")

    al = filtrar_por_insumo(alertas_df, insumos_sel) if not alertas_df.empty else alertas_df
    if not al.empty:
        n_falta = int((al["status"] == "FALTA").sum())
        n_excesso = int((al["status"] == "EXCESSO").sum())
        capital = float(al["custo_excesso"].sum()) if "custo_excesso" in al.columns else 0.0
        linhas.append("--- Alertas de estoque ---")
        linhas.append(
            f"Insumos em FALTA: {n_falta} | em EXCESSO: {n_excesso} | "
            f"Capital em risco (excesso): R$ {capital:,.2f}"
        )
        cols_al = [
            c for c in [
                "insumo", "unidade", "estoque_atual", "necessidade_total",
                "saldo", "status", "custo_excesso",
            ] if c in al.columns
        ]
        linhas.append(_df_para_texto(al[cols_al].sort_values("status")))
        linhas.append("")

    if saldos_dict:
        df_s = pd.DataFrame(
            [{"insumo": k, "saldo_atual": v} for k, v in saldos_dict.items()]
        )
        df_s = df_s.merge(
            insumos_info[["nome", "unidade", "custo_unitario"]],
            left_on="insumo", right_on="nome", how="left",
        )
        df_s = filtrar_por_insumo(df_s, insumos_sel)
        df_s["valor"] = (df_s["saldo_atual"] * df_s["custo_unitario"]).round(2)
        linhas.append("--- Saldos atuais de estoque ---")
        linhas.append(f"Valor total em estoque: R$ {df_s['valor'].sum():,.2f}")
        linhas.append(
            _df_para_texto(
                df_s[["insumo", "saldo_atual", "unidade", "custo_unitario", "valor"]]
                .sort_values("valor", ascending=False)
            )
        )
        linhas.append("")

    nd = (
        filtrar_por_insumo(necessidade_df, insumos_sel)
        if not necessidade_df.empty else necessidade_df
    )
    if not nd.empty and "quantidade_necessaria" in nd.columns:
        tot = (
            nd.groupby("insumo", observed=True)["quantidade_necessaria"]
            .sum().sort_values(ascending=False).reset_index()
        )
        linhas.append(f"--- Necessidade total de insumos nos próximos {H} dias ---")
        linhas.append(_df_para_texto(tot.head(20)))
        linhas.append("")

    v = (
        filtrar_vendas(vendas_df, data_ini, data_fim, pratos_sel)
        if not vendas_df.empty else vendas_df
    )
    if not v.empty:
        por_prato = (
            v.groupby("prato", observed=True)["quantidade"]
            .agg(["sum", "mean"]).round(1)
            .rename(columns={"sum": "total", "mean": "media_dia"})
            .sort_values("total", ascending=False).reset_index()
        )
        linhas.append("--- Vendas no período filtrado (por prato) ---")
        linhas.append(_df_para_texto(por_prato))
        linhas.append(f"Total de pratos vendidos no período: {int(v['quantidade'].sum())}")
        linhas.append("")

    if kpis_desperdicio:
        linhas.append("--- Desperdício ---")
        linhas.append(
            f"Taxa de desperdício: {kpis_desperdicio.get('taxa_desperdicio_pct', 0):.2f}% | "
            f"Custo total de perdas: R$ {kpis_desperdicio.get('custo_total_perdas', 0):,.2f} | "
            f"Registros de perda: {kpis_desperdicio.get('n_registros_perda', 0)}"
        )
        df_motivo = load_csv("desperdicio_por_motivo.csv")
        if not df_motivo.empty:
            if "insumo" in df_motivo.columns:
                df_motivo = filtrar_por_insumo(df_motivo, insumos_sel)
            linhas.append("Perdas por motivo:")
            linhas.append(_df_para_texto(df_motivo.head(10)))
        linhas.append("")

    try:
        from src.predict import prever_proximos_dias
        prev = prever_proximos_dias(H, inicio_prev)
        if not prev.empty:
            prev = prev[prev["prato"].isin(pratos_sel)]
        if not prev.empty:
            tot_prev = (
                prev.groupby("prato", observed=True)["previsao"]
                .sum().round(0).sort_values(ascending=False).reset_index()
            )
            linhas.append(f"--- Previsão de demanda ({H} dias) — total por prato ---")
            linhas.append(_df_para_texto(tot_prev))
            linhas.append("")
    except Exception:
        pass

    linhas.append("=== FIM DO CONTEXTO ===")
    linhas.append(
        "Instruções: responda SEMPRE em português do Brasil. Use apenas os números "
        "e informações acima. Seja objetivo e acionável (sugira compras, alertas, "
        "ações práticas). Não invente dados. Se faltar informação, diga claramente."
    )
    return "\n".join(linhas)


def _obter_chave_groq() -> str | None:
    """Tenta pegar a chave da API do Groq (secrets ou variável de ambiente)."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_TOKEN")

with aba_chat:
    st.subheader("Assistente de análise")
    st.caption(
        "Pergunte sobre estoque, vendas, alertas, desperdício, necessidade de compra "
        "ou previsão de demanda. As respostas usam os dados atuais do sistema "
        "(respeitando os filtros da barra lateral)."
    )

    api_key = _obter_chave_groq()

    if not api_key:
        st.warning(
            "Para usar o assistente, configure a chave gratuita do **Groq**:\n\n"
            "1. Crie uma conta em https://console.groq.com e gere uma API Key\n"
            "2. Coloque a chave em um dos lugares:\n"
            "   - arquivo `.streamlit/secrets.toml` → `GROQ_API_KEY = \"gsk_...\"`\n"
            "   - ou variável de ambiente: `export GROQ_API_KEY=gsk_...`\n\n"
            "O Groq tem tier gratuito generoso e é bem rápido — ideal para protótipo."
        )
        st.stop()

    st.markdown(
        """
        <style>
        div[data-testid="stChatMessage"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background-color: rgba(128, 128, 128, 0.08);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.3rem;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background-color: rgba(74, 144, 217, 0.10);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Olá! Sou o assistente do restaurante. "
                    "Posso analisar estoque, vendas, alertas, desperdício e previsão de demanda "
                    "com base nos dados atuais (e nos filtros da barra lateral).\n\n"
                    "Pergunte o que quiser, ou use uma sugestão abaixo."
                ),
            }
        ]

    # ---------- área de mensagens com scroll ----------
    with st.container(height=420, border=True):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # força o scroll para o final da área de chat
        st.markdown(
            """
            <script>
            const boxes = window.parent.document.querySelectorAll(
                'div[data-testid="stVerticalBlockBorderWrapper"]'
            );
            if (boxes.length) {
                const box = boxes[boxes.length - 1];
                box.scrollTop = box.scrollHeight;
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

    # ---------- sugestões + limpar ----------
    sugestoes = [
        "Quais insumos estão em falta?",
        "Quanto devo comprar agora?",
        "Resumo do desperdício",
        "Previsão de vendas por prato",
        "O que está em excesso no estoque?",
    ]

    c_sug = st.columns(len(sugestoes) + 1)
    for i, s in enumerate(sugestoes):
        if c_sug[i].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state["_pergunta_pendente"] = s
            st.rerun()
    if c_sug[-1].button("🗑️ Limpar", key="btn_limpar_chat", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversa limpa. Em que posso ajudar com estoque, vendas ou desperdício?"
                ),
            }
        ]
        st.session_state.pop("_pergunta_pendente", None)
        st.rerun()

    # ---------- input SEMPRE visível ----------
    pergunta_digitada = st.chat_input(
        "Digite sua pergunta sobre estoque, vendas, desperdício..."
    )

    # prioriza botão de sugestão; senão usa o que foi digitado
    pergunta = st.session_state.pop("_pergunta_pendente", None) or pergunta_digitada

    if pergunta:
        st.session_state.chat_messages.append({"role": "user", "content": pergunta})

        try:
            from groq import Groq

            contexto = _construir_contexto_ia(
                vendas_df=vendas,
                alertas_df=alertas,
                necessidade_df=necessidade_diaria,
                saldos_dict=saldos,
                insumos_info=insumos_df,
                kpis_desperdicio=taxa_desperdicio(),
                data_ini=data_ini,
                data_fim=data_fim,
                pratos_sel=pratos_sel,
                insumos_sel=insumos_sel,
                H=H,
                inicio_prev=INICIO_PREV,
                fim_prev=FIM_PREV,
            )

            client = Groq(api_key=api_key)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Você é o assistente inteligente de um restaurante. "
                        "Ajuda o gestor a reduzir desperdício, planejar compras e "
                        "entender demanda. Responda sempre em português do Brasil, "
                        "de forma clara, objetiva e prática. Baseie-se EXCLUSIVAMENTE "
                        "nos dados fornecidos no contexto. Não invente números. "
                        "Use listas e destaques quando ajudar a decisão."
                    ),
                },
                {"role": "system", "content": contexto},
            ]
            for m in st.session_state.chat_messages[-8:]:
                messages.append({"role": m["role"], "content": m["content"]})

            with st.spinner("Analisando os dados..."):
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                )
            resposta = completion.choices[0].message.content.strip()
        except Exception as e:
            resposta = (
                f"Não foi possível obter resposta da IA.\n\n"
                f"Erro: `{e}`\n\n"
                "Verifique se a chave GROQ_API_KEY está correta e se há "
                "crédito/quota disponível no console do Groq."
            )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": resposta}
        )
        st.rerun()  # redesenha histórico + mantém o input