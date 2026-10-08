import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import subprocess
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import timedelta

from src.config import PROCESSED_DIR, MODELS_DIR
from src.data_loader import carregar_vendas
from src.waste import taxa_desperdicio
from src.data_loader import carregar_vendas, carregar_insumos
from datetime import date, timedelta
from src.config import HORIZONTE_PREVISAO

def filtrar_vendas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        (df["data"].dt.date >= data_ini)
        & (df["data"].dt.date <= data_fim)
        & (df["prato"].isin(pratos_sel))
    )
    return df[mask].copy()


def filtrar_por_insumo(df: pd.DataFrame, col: str = "insumo") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    return df[df[col].isin(insumos_sel)].copy()

st.set_page_config(
    page_title="Previsão de Demanda — Restaurante",
    page_icon="🍽️",
    layout="wide",
)

def _reprocessar(horizonte: int | None = None):
    """Re-executa o pipeline de negócio e limpa o cache do Streamlit."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if horizonte is not None:
        env["HORIZONTE_PREVISAO"] = str(horizonte)
    subprocess.run(
        [sys.executable, "-m", "src.business"],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    st.cache_data.clear()
    st.cache_resource.clear()

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
        _reprocessar()
        st.success("Movimentação registrada.")
        st.rerun()
    if c2.button("❌ Cancelar", use_container_width=True):
        st.rerun()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# FILTROS GLOBAIS (sidebar)
# ------------------------------------------------------------------
st.sidebar.subheader("Filtros globais")

# --- Período
vendas_full = load_vendas()

if not vendas_full.empty:
    data_min = vendas_full["data"].min().date()
    data_max = vendas_full["data"].max().date()

    # Contador de versão para forçar reset quando usar atalhos
    if "periodo_versao" not in st.session_state:
        st.session_state["periodo_versao"] = 0
    if "periodo_default" not in st.session_state:
        st.session_state["periodo_default"] = (data_min, data_max)

    periodo = st.sidebar.date_input(
        "Período",
        value=st.session_state["periodo_default"],
        min_value=data_min,
        max_value=data_max,
        key=f"filtro_periodo_v{st.session_state['periodo_versao']}",
    )

    if isinstance(periodo, tuple) and len(periodo) == 2:
        data_ini, data_fim = periodo
    else:
        data_ini, data_fim = data_min, data_max
else:
    data_ini, data_fim = None, None
    data_min, data_max = None, None

# --- Pratos
todos_pratos = sorted(vendas_full["prato"].unique().tolist()) if not vendas_full.empty else []
pratos_sel = st.sidebar.multiselect(
    "Pratos",
    todos_pratos,
    default=todos_pratos,
    key="filtro_pratos",
)

# --- Insumos
insumos_full = carregar_insumos()
todos_insumos = sorted(insumos_full["nome"].unique().tolist())
insumos_sel = st.sidebar.multiselect(
    "Insumos",
    todos_insumos,
    default=todos_insumos,
    key="filtro_insumos",
)
st.sidebar.divider()
st.sidebar.subheader("Horizonte de previsão")

if "horizonte_atual" not in st.session_state:
    st.session_state["horizonte_atual"] = HORIZONTE_PREVISAO

H = st.sidebar.slider(
    "Dias à frente",
    min_value=2,
    max_value=30,
    value=st.session_state["horizonte_atual"],
    step=1,
    key="slider_horizonte",
)

if H != st.session_state["horizonte_atual"]:
    st.session_state["horizonte_atual"] = H
    with st.spinner(f"Recalculando para {H} dias..."):
        _reprocessar(horizonte=H)
    st.rerun()


# --- Atalhos rápidos
st.sidebar.caption("Atalhos rápidos")


def _aplicar_atalho(dias: int | None = None):
    if dias is None:
        novo_ini = data_min
    else:
        novo_ini = max(data_min, data_max - timedelta(days=dias - 1))
    st.session_state["periodo_default"] = (novo_ini, data_max)
    st.session_state["periodo_versao"] += 1


c1, c2, c3 = st.sidebar.columns(3)
c1.button("7d",  use_container_width=True, on_click=_aplicar_atalho, args=(7,))
c2.button("30d", use_container_width=True, on_click=_aplicar_atalho, args=(30,))
c3.button("Tudo", use_container_width=True, on_click=_aplicar_atalho, args=(None,))
if st.sidebar.button("🔄 Rodar pipeline completo"):
    with st.spinner("Gerando dados, treinando e calculando alertas..."):
        for cmd in [
            [sys.executable, "-m", "src.data_generator"],
            [sys.executable, "-m", "src.train"],
            [sys.executable, "-m", "src.business"],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                st.sidebar.error(f"Erro em {cmd[-1]}:\n{r.stderr[-500:]}")
                st.stop()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Pipeline executado.")
    st.rerun()
if st.sidebar.button("🔄 Regerar Tudo"):
    with st.spinner("Gerando dados, treinando e calculando alertas..."):
        for cmd in [
            [sys.executable, "-m", "src.data_generator"],
            [sys.executable, "-m", "src.train"],
            [sys.executable, "-m", "src.inventory"],
            [sys.executable, "-m", "src.business"],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                st.sidebar.error(f"Erro em {cmd[-1]}:\n{r.stderr[-500:]}")
                st.stop()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Pipeline executado.")
    st.rerun()


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🍽️ Previsão de Demanda e Redução de Desperdício")
st.caption("Projeto TG — Machine Learning aplicado ao estoque de restaurante")

vendas = load_vendas()
alertas = load_csv("alertas.csv")
comparacao = load_csv("comparacao_estoque.csv")
necessidade_diaria = load_csv("necessidade_insumos_diaria.csv")

if vendas.empty or alertas.empty:
    st.warning("Rode o pipeline primeiro (botão na lateral).")
    st.stop()


# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
pacote = load_modelo()
wape_final = None
if pacote and "tabela_metricas" in pacote:
    tab = pacote["tabela_metricas"]
    if "ensemble_media_ponderada" in tab.index:
        wape_final = tab.loc["ensemble_media_ponderada", "WAPE (%)"]
    else:
        wape_final = tab["WAPE (%)"].min()

alertas_f = filtrar_por_insumo(alertas)
n_falta = int((alertas_f["status"] == "FALTA").sum())
n_excesso = int((alertas_f["status"] == "EXCESSO").sum())
capital_risco = float(alertas_f["custo_excesso"].sum())
custo_compra = float(
    (alertas_f.loc[alertas_f["status"] == "FALTA", "saldo"].abs()
     * alertas_f.loc[alertas_f["status"] == "FALTA", "custo_unitario"]).sum()
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WAPE (ensemble)", f"{wape_final:.2f}%" if wape_final else "—")
c2.metric("Insumos em falta", n_falta)
c3.metric("Insumos com excesso", n_excesso)
c4.metric("Compra sugerida", f"R$ {custo_compra:,.2f}")
c5.metric("Capital em risco", f"R$ {capital_risco:,.2f}")

st.caption(
    f"Filtros ativos — período: {data_ini} a {data_fim} | "
    f"{len(pratos_sel)}/{len(todos_pratos)} pratos | "
    f"{len(insumos_sel)}/{len(todos_insumos)} insumos"
)


# ------------------------------------------------------------------
# ABAS
# ------------------------------------------------------------------
aba_hist, aba_vendas, aba_modelo, aba_prev, aba_insumos, aba_alertas, aba_estoque, aba_desp, aba_cad = st.tabs([
    "📊 Histórico",
    "🧾 Vendas",
    "🤖 Modelo",
    "🔮 Previsão",
    "🥕 Insumos",
    "🚨 Alertas",
    "📦 Estoque",
    "🗑️ Desperdício",
    "📋 Cadastros",
])


# ---- Histórico
with aba_hist:
    st.subheader("Vendas históricas")
    v = filtrar_vendas(vendas)

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
        fmt_eixo = {"Diária": "%d/%m", "Semanal": "S%W/%y", "Mensal": "%b/%y"}[agregacao]

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
        fig_total = px.line(total, x="data", y="total")
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
        st.plotly_chart(fig_total, use_container_width=True)

        # ---------- Gráfico 2: composição ----------
        st.markdown("**Composição por prato**")
        fig_area = px.area(
            por_prato, x="data", y="quantidade", color="prato",
        )
        fig_area.update_xaxes(tickformat=fmt_eixo, nticks=8)
        fig_area.update_yaxes(rangemode="tozero", title="Pratos vendidos")
        fig_area.update_layout(
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_area, use_container_width=True)

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
            media_dow, x="dia_semana_pt", y="quantidade",
            color="prato", barmode="group",
            category_orders={"dia_semana_pt": ordem_pt},
        )
        fig2.update_xaxes(title="")
        fig2.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ---------- Feriado vs. dia comum ----------
        st.subheader("Comparação feriado vs. dia comum")
        v2["tipo_dia"] = v2["feriado"].map({0: "Dia comum", 1: "Feriado"})
        media_fer = (
            v2.groupby(["tipo_dia", "prato"], observed=True)["quantidade"]
              .mean()
              .reset_index()
        )
        fig3 = px.bar(
            media_fer, x="tipo_dia", y="quantidade",
            color="prato", barmode="group",
        )
        fig3.update_xaxes(title="")
        fig3.update_layout(
            height=380,
            legend=dict(orientation="h", y=-0.25),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Ver tabela bruta"):
            st.dataframe(
                v.sort_values("data", ascending=False),
                use_container_width=True,
            )

with aba_vendas:
    from src.catalog import listar_pratos, listar_vendas, registrar_venda, remover_venda, ficha_do_prato
    from src.inventory import registrar_saida_por_venda, saldo_atual
    from src.data_loader import carregar_insumos

    st.subheader("Registrar venda")
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
                                placeholder="Selecione...", key="venda_prato_sel")
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
                st.info(f"'{prato_v}' não tem ficha técnica cadastrada. "
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
                st.dataframe(
                    preview[["insumo", "quantidade", "estoque_atual",
                             "saldo_pos", "ok"]].round(3),
                    use_container_width=True,
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
                            f"{item['insumo']}: necessário {necessario:.2f}, "
                            f"disponível {disponivel:.2f}"
                        )
                if faltando:
                    erros.append(
                        "Estoque insuficiente. Ajuste a quantidade, registre uma "
                        "entrada/ajuste na aba Estoque, ou desmarque a baixa automática."
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
                        f"{int(qtd_v)}x {prato_v} registrado em {data_v}."
                    )
                    st.rerun()
                except Exception as e:
                    st.session_state["msg_venda"] = ("erro", str(e))
                    st.rerun()

    st.divider()
    st.subheader("Vendas registradas")

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
        )

        filtro = vendas_df[
            (vendas_df["data"] >= f_ini)
            & (vendas_df["data"] <= f_fim)
            & (vendas_df["prato"].isin(f_prato))
        ].sort_values("data", ascending=False)

        m1, m2, m3 = st.columns(3)
        m1.metric("Registros", len(filtro))
        m2.metric("Total de pratos vendidos", int(filtro["quantidade"].sum()))
        m3.metric("Média por registro",
                  f"{filtro['quantidade'].mean():.1f}" if len(filtro) else "—")

        st.dataframe(filtro.head(500), use_container_width=True)

        st.divider()
        with st.expander("🗑️ Remover venda"):
            c1, c2 = st.columns(2)
            pra_r = c1.selectbox("Prato",
                                  sorted(vendas_df["prato"].unique().tolist()),
                                  key="rm_venda_prato")
            datas_disp = sorted(vendas_df[vendas_df["prato"] == pra_r]["data"].unique())
            data_r = c2.selectbox("Data", datas_disp, key="rm_venda_data")
            st.caption(
                "⚠️ Remover a venda **não** estorna o consumo do estoque. "
                "Se quiser estornar, use **📦 Estoque → Ajuste de auditoria**."
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
        st.dataframe(
            pacote["tabela_metricas"].style.highlight_min(
                axis=0, subset=["MAE", "RMSE", "WAPE (%)"], color="#87A96B"
            ),
            use_container_width=True,
        )
        melhor = pacote.get("melhor_modelo", "—")
        wape_melhor = pacote["tabela_metricas"].loc[melhor, "WAPE (%)"] if melhor in pacote["tabela_metricas"].index else None
        msg = f"Modelo final: **{melhor}**"
        if wape_melhor is not None:
            msg += f" — WAPE {wape_melhor:.2f}%"
        st.success(msg)
    else:
        st.info("Treine o modelo para visualizar as métricas.")


# ---- Previsão
with aba_prev:
    from src.predict import prever_proximos_dias

    H = st.session_state["horizonte_atual"]

    st.subheader(f"Previsão dos próximos {H} dias")
    st.caption(
        "Use o controle **Horizonte de previsão** na barra lateral "
        "para alterar o número de dias previstos."
    )

    c1, c2 = st.columns([3, 1])
    with c2:
        tipo_vis = st.radio("Visualização", ["Linha", "Área", "Barras"],
                             index=0, key="prev_vis")

    try:
        prev = prever_proximos_dias(H)
        prev_f = prev[prev["prato"].isin(pratos_sel)]

        if prev_f.empty:
            st.warning("Nenhum prato no filtro atual.")
        else:
            fig_kwargs = dict(
                x="data", y="previsao", color="prato",
                title=f"Previsão — próximos {H} dias",
            )
            if tipo_vis == "Linha":
                fig3 = px.line(prev_f, markers=True, **fig_kwargs)
            elif tipo_vis == "Área":
                fig3 = px.area(prev_f, **fig_kwargs)
            else:
                fig3 = px.bar(prev_f, barmode="group", **fig_kwargs)
            st.plotly_chart(fig3, use_container_width=True)

            # Histórico + previsão
            v = filtrar_vendas(vendas)
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
                combo, x="data", y="quantidade", color="prato",
                line_dash="tipo",
                title="Últimos 30 dias + previsão",
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown(f"**Tabela de previsão — {H} dias**")
            st.dataframe(
                prev_f.pivot(index="data", columns="prato", values="previsao"),
                use_container_width=True,
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
            st.dataframe(totais, use_container_width=True)

    except Exception as e:
        st.error(f"Não foi possível gerar previsão: {e}")


# ---- Insumos
with aba_insumos:
    H = st.session_state["horizonte_atual"]
    st.subheader(f"Necessidade diária de insumos — próximos {H} dias")

    nd = filtrar_por_insumo(necessidade_diaria)
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
            nd, x="data", y=col_y, color="insumo", barmode="stack",
            category_orders={"insumo": ordem_insumos},
            title=f"Consumo previsto por dia — {metrica}",
        )
        fig4.update_yaxes(title=ylabel)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Tabela de necessidade diária**")
        pivot = nd.pivot_table(
            index="data", columns="insumo",
            values=col_y, aggfunc="sum",
        ).round(2)
        st.dataframe(pivot, use_container_width=True)

        st.markdown("**Total por insumo no período**")
        total = (
            nd.groupby("insumo", observed=True)[col_y].sum()
              .sort_values(ascending=False).round(2).reset_index()
        )
        st.dataframe(total, use_container_width=True)


# ---- Alertas
with aba_alertas:
    H = st.session_state["horizonte_atual"]
    st.subheader(f"Status dos insumos — próximos {H} dias")
    st.caption(
        f"A **necessidade** de cada insumo é calculada a partir da previsão "
        f"de demanda dos próximos {H} dias, convertida via ficha técnica. "
        f"O **saldo** é a diferença entre o estoque atual e essa necessidade."
    )

    al = filtrar_por_insumo(alertas)

    c1, c2 = st.columns([1, 1])
    with c2:
        status_sel = st.multiselect(
            "Status",
            ["FALTA", "OK", "EXCESSO"],
            default=["FALTA", "OK", "EXCESSO"],
            key="al_status",
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
            "FALTA": "background-color: #E53E3E",
            "EXCESSO": "background-color: #FFB300",
            "OK": "background-color: #87A96B",
        }.get(s, "")

    cols_show = ["insumo", "unidade", "estoque_atual", "necessidade_total",
                 "saldo", "status", "custo_excesso"]
    rename = {
        "necessidade_total": f"necessidade_{H}d",
        "saldo": f"saldo_{H}d",
        "custo_excesso": "custo_excesso_R$",
    }
    tabela_mostrar = al[cols_show].rename(columns=rename)
    st.dataframe(
        tabela_mostrar.style.map(cor_status, subset=["status"]),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Sugestão de compra")
    falta = al[al["status"] == "FALTA"].copy()
    if falta.empty:
        st.success("Nenhuma compra necessária no filtro atual.")
    else:
        falta["qtd_sugerida"] = (-falta["saldo"] * 1.10).round(2)
        falta["custo_estimado"] = (falta["qtd_sugerida"] * falta["custo_unitario"]).round(2)
        st.dataframe(
            falta[["insumo", "unidade", "qtd_sugerida",
                   "custo_unitario", "custo_estimado"]],
            use_container_width=True,
        )
        st.info(f"Total estimado: **R$ {falta['custo_estimado'].sum():,.2f}**")

        fig_c = px.bar(falta, x="insumo", y="custo_estimado",
                       title="Custo de compra por insumo (R$)")
        st.plotly_chart(fig_c, use_container_width=True)

    if not al[al["status"] == "EXCESSO"].empty:
        st.divider()
        st.subheader("⚠️ Insumos com excesso (risco de desperdício)")
        exc = al[al["status"] == "EXCESSO"]
        st.warning(
            f"{len(exc)} insumo(s) com estoque acima do necessário. "
            f"Capital em risco: **R$ {exc['custo_excesso'].sum():,.2f}**."
        )
    
with aba_estoque:
    from src.inventory import (
        saldos_atuais, todas_movimentacoes,
        registrar_entrada, registrar_perda, registrar_ajuste,
    )
    from src.data_loader import carregar_insumos

    insumos_df = carregar_insumos()
    lista_insumos = insumos_df["nome"].tolist()
    unidade_de = dict(zip(insumos_df["nome"], insumos_df["unidade"]))

    # ---------- Saldos atuais ----------
    st.subheader("Saldos atuais")
    saldos = saldos_atuais()
    df_saldos = pd.DataFrame(
        [{"insumo": k, "saldo_atual": v} for k, v in saldos.items()]
    ).merge(insumos_df, left_on="insumo", right_on="nome", how="left")
    df_saldos["valor_total"] = (
        df_saldos["saldo_atual"] * df_saldos["custo_unitario"]
    ).round(2)
    st.dataframe(
        df_saldos[["insumo", "saldo_atual", "unidade",
                   "custo_unitario", "valor_total"]],
        use_container_width=True,
    )

    st.divider()
    st.subheader("Registrar movimentação")

    # ---------- ENTRADA ----------
    with st.expander("➕ Entrada de insumo", expanded=False):
        with st.form("form_entrada", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            insumo_e = c1.selectbox("Insumo", lista_insumos, index=None,
                                     placeholder="Selecione...", key="ent_insumo")
            unidade_e = unidade_de.get(insumo_e, "un") if insumo_e else "un"
            qtd_e = c2.number_input(f"Quantidade ({unidade_e})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="ent_qtd")
            custo_e = c3.number_input("Custo unitário (R$)",
                                       min_value=0.0, step=0.5,
                                       value=None, key="ent_custo")

            c4, c5 = st.columns(2)
            origem_e = c4.text_input("Fornecedor / origem",
                                      value="", key="ent_origem")
            obs_e = c5.text_input("Observação", value="", key="ent_obs")

            submitted = st.form_submit_button("Registrar entrada")

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
                        "Insumo": insumo_e,
                        "Quantidade": f"{qtd_e} {unidade_e}",
                        "Custo unitário": f"R$ {custo_e:.2f}",
                        "Origem": origem_e or "—",
                        "Observação": obs_e or "—",
                    },
                    lambda: registrar_entrada(
                        insumo_e, qtd_e, custo_e,
                        origem_e or "fornecedor", obs_e,
                    ),
                )

    # ---------- PERDA ----------
    with st.expander("🗑️ Registrar perda / desperdício", expanded=False):
        with st.form("form_perda", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            insumo_p = c1.selectbox("Insumo", lista_insumos, index=None,
                                     placeholder="Selecione...", key="per_insumo")
            unidade_p = unidade_de.get(insumo_p, "un") if insumo_p else "un"
            qtd_p = c2.number_input(f"Quantidade ({unidade_p})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="per_qtd")
            motivo_p = c3.selectbox("Motivo",
                                     ["validade", "preparo", "sobra", "avaria", "outro"],
                                     index=None, placeholder="Selecione...",
                                     key="per_motivo")

            obs_p = st.text_input("Observação", value="", key="per_obs")
            submitted = st.form_submit_button("Registrar perda")

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
                        "Insumo": insumo_p,
                        "Quantidade": f"{qtd_p} {unidade_p}",
                        "Motivo": motivo_p,
                        "Observação": obs_p or "—",
                    },
                    lambda: registrar_perda(insumo_p, qtd_p, motivo_p, obs_p),
                )

    # ---------- AJUSTE ----------
    with st.expander("🧮 Ajuste de auditoria (contagem física)", expanded=False):
        with st.form("form_ajuste", clear_on_submit=True):
            c1, c2 = st.columns(2)
            insumo_a = c1.selectbox("Insumo", lista_insumos, index=None,
                                     placeholder="Selecione...", key="aj_insumo")
            unidade_a = unidade_de.get(insumo_a, "un") if insumo_a else "un"
            saldo_a = saldos.get(insumo_a, 0.0) if insumo_a else 0.0
            c2.metric("Saldo no sistema", f"{saldo_a:.2f} {unidade_a}")
            qtd_a = st.number_input(f"Quantidade contada ({unidade_a})",
                                     min_value=0.0, step=0.5,
                                     value=None, key="aj_qtd")
            obs_a = st.text_input("Observação", value="", key="aj_obs")
            submitted = st.form_submit_button("Registrar ajuste")

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
                        "Insumo": insumo_a,
                        "Saldo no sistema": f"{saldo_a:.2f} {unidade_a}",
                        "Contagem física": f"{qtd_a:.2f} {unidade_a}",
                        "Diferença": f"{delta:+.2f} {unidade_a}",
                        "Observação": obs_a or "—",
                    },
                    lambda: registrar_ajuste(insumo_a, qtd_a, obs_a),
                )

    st.divider()
    st.subheader("Histórico de movimentações")
    movs = todas_movimentacoes()

    if movs.empty:
        st.info("Nenhuma movimentação registrada ainda.")
    else:
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        f_tipo = c1.multiselect(
            "Tipo", sorted(movs["tipo"].unique().tolist()),
            default=sorted(movs["tipo"].unique().tolist()),
        )
        f_insumo = c2.multiselect(
            "Insumo", sorted(movs["insumo"].unique().tolist()),
            default=[i for i in sorted(movs["insumo"].unique().tolist())
                     if i in insumos_sel],
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
        st.dataframe(
            filtro.sort_values("data", ascending=False).head(int(limite)),
            use_container_width=True,
            column_config={
                "data": st.column_config.DatetimeColumn(
                    "data",
                    format="DD/MM/YYYY HH:mm",
                ),
            },
        )
        st.caption(f"{len(filtro)} movimentações no filtro atual.")

with aba_desp:
    st.subheader("Análise de desperdício")

    kpis = taxa_desperdicio()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taxa de desperdício", f"{kpis['taxa_desperdicio_pct']:.2f}%")
    c2.metric("Custo total de perdas", f"R$ {kpis['custo_total_perdas']:,.2f}")
    c3.metric("Custo total de entradas", f"R$ {kpis['custo_total_entradas']:,.2f}")
    c4.metric("Registros de perda", kpis["n_registros_perda"])

    st.caption(
        "A taxa de desperdício representa o percentual do valor comprado "
        "que foi perdido. Valores acima de 10% indicam oportunidade de melhoria."
    )

    st.divider()

    df_motivo = filtrar_por_insumo(load_csv("desperdicio_por_motivo.csv"))
    df_insumo = filtrar_por_insumo(load_csv("desperdicio_por_insumo.csv"))
    df_dia = load_csv("desperdicio_por_dia.csv")
    df_balanco = filtrar_por_insumo(load_csv("balanco_insumos.csv"))
    df_ctr = filtrar_por_insumo(load_csv("consumo_teorico_vs_real.csv"))

    if df_dia.empty:
        st.info(
            "Nenhuma perda registrada ainda. Use a aba **Estoque** para "
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
                fig_m = px.pie(df_motivo, names="motivo", values=col_v,
                               hole=0.4,
                               title=f"Distribuição por motivo — {metrica_d}")
                st.plotly_chart(fig_m, use_container_width=True)

        with c2:
            st.markdown("**Top insumos por perda**")
            if not df_insumo.empty:
                top_n = st.slider("Top N", 3, 15, 10, key="d_topn")
                fig_i = px.bar(
                    df_insumo.head(top_n), x="custo", y="insumo",
                    orientation="h",
                    title=f"Top {top_n} insumos por custo de perda",
                )
                fig_i.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_i, use_container_width=True)

        if not df_dia.empty:
            st.markdown("**Evolução diária das perdas**")
            df_dia["data"] = pd.to_datetime(df_dia["data"])
            df_dia_f = df_dia[
                (df_dia["data"].dt.date >= data_ini)
                & (df_dia["data"].dt.date <= data_fim)
            ]
            if not df_dia_f.empty:
                fig_d = px.line(df_dia_f, x="data", y="custo", markers=True,
                                title="Custo de perdas por dia (R$)")
                st.plotly_chart(fig_d, use_container_width=True)

        st.divider()
        st.subheader("Consumo teórico vs. real")
        st.caption(
            "Teórico = vendas × ficha técnica. Real = saídas por venda. "
            "Diferença positiva indica desperdício oculto ou erro de porção."
        )
        if not df_ctr.empty:
            fig_ctr = px.bar(
                df_ctr, x="insumo", y=["teorico", "saida_real"],
                barmode="group",
                title="Consumo teórico vs. real (kg)",
            )
            st.plotly_chart(fig_ctr, use_container_width=True)
            st.dataframe(df_ctr, use_container_width=True)

        st.divider()
        st.subheader("Balanço por insumo")
        st.caption(
            "Verificação contábil: **comprado = consumido + perdido + saldo** "
            "(± ajustes)."
        )
        if not df_balanco.empty:
            st.dataframe(df_balanco, use_container_width=True)

with aba_cad:
    from src.catalog import (
        listar_insumos, adicionar_insumo, editar_insumo, remover_insumo,
        listar_pratos, adicionar_prato, editar_prato, remover_prato,
        listar_ficha, ficha_do_prato, adicionar_item_ficha, remover_item_ficha,
        listar_vendas, registrar_venda, remover_venda,
    )
    from src.inventory import registrar_saida_por_venda

    st.subheader("Cadastros")
    st.caption(
        "Alterações aqui afetam o pipeline inteiro. Depois de editar, "
        "recomendo rodar **🔄 Rodar pipeline completo** para retreinar o modelo."
    )

    sub_ins, sub_pra, sub_fic = st.tabs(
        ["🥕 Insumos", "🍽️ Pratos", "📖 Ficha técnica"]
    )

    # --------------------------------------------------------------
    # INSUMOS
    # --------------------------------------------------------------
    with sub_ins:
        st.markdown("**Insumos cadastrados**")
        ins = listar_insumos()
        st.dataframe(ins, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            with st.form("form_add_insumo", clear_on_submit=True):
                st.markdown("**Adicionar insumo**")
                nome_i = st.text_input("Nome", value="")
                unidade_i = st.selectbox("Unidade", ["kg", "g", "L", "mL", "un"])
                custo_i = st.number_input("Custo unitário (R$)", min_value=0.0,
                                           step=0.5, value=None)
                if st.form_submit_button("Adicionar"):
                    try:
                        adicionar_insumo(nome_i, unidade_i, custo_i or 0.0)
                        st.success(f"Insumo '{nome_i}' adicionado.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with c2:
            with st.form("form_edit_insumo", clear_on_submit=True):
                st.markdown("**Editar insumo**")
                nomes = ins["nome"].tolist() if not ins.empty else []
                nome_e = st.selectbox("Insumo", nomes) if nomes else None
                unidade_e = st.selectbox("Nova unidade", ["kg", "g", "L", "mL", "un"],
                                          key="edit_uni")
                custo_e = st.number_input("Novo custo (R$)", min_value=0.0,
                                           step=0.5, value=None, key="edit_custo")
                if st.form_submit_button("Salvar") and nome_e:
                    try:
                        atual = ins[ins["nome"] == nome_e].iloc[0]
                        editar_insumo(
                            nome_e,
                            unidade_e,
                            custo_e if custo_e is not None else float(atual["custo_unitario"]),
                        )
                        st.success("Insumo atualizado.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.divider()
        st.markdown("**Remover insumo**")
        st.caption("Só é possível remover insumos que não estejam em nenhuma ficha técnica.")
        nome_r = st.selectbox("Insumo a remover",
                               ins["nome"].tolist() if not ins.empty else [],
                               key="rm_insumo")
        if st.button("🗑️ Remover insumo", key="btn_rm_insumo") and nome_r:
            try:
                remover_insumo(nome_r)
                st.success(f"Insumo '{nome_r}' removido.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # --------------------------------------------------------------
    # PRATOS
    # --------------------------------------------------------------
    with sub_pra:
        st.markdown("**Pratos cadastrados**")
        pra = listar_pratos()
        st.dataframe(pra, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            with st.form("form_add_prato", clear_on_submit=True):
                st.markdown("**Adicionar prato**")
                nome_p = st.text_input("Nome do prato", value="")
                if st.form_submit_button("Adicionar"):
                    try:
                        adicionar_prato(nome_p)
                        st.success(f"Prato '{nome_p}' adicionado.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with c2:
            with st.form("form_edit_prato", clear_on_submit=True):
                st.markdown("**Renomear prato**")
                nomes_p = pra["nome"].tolist() if not pra.empty else []
                antigo = st.selectbox("Prato", nomes_p) if nomes_p else None
                novo = st.text_input("Novo nome", value="")
                if st.form_submit_button("Renomear") and antigo and novo:
                    try:
                        editar_prato(antigo, novo)
                        st.success(f"'{antigo}' renomeado para '{novo}'.")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        st.divider()
        st.markdown("**Remover prato**")
        st.caption("Só é possível remover pratos sem ficha técnica e sem vendas registradas.")
        nome_rp = st.selectbox("Prato a remover",
                                pra["nome"].tolist() if not pra.empty else [],
                                key="rm_prato")
        if st.button("🗑️ Remover prato", key="btn_rm_prato") and nome_rp:
            try:
                remover_prato(nome_rp)
                st.success(f"Prato '{nome_rp}' removido.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # --------------------------------------------------------------
    # FICHA TÉCNICA
    # --------------------------------------------------------------
    with sub_fic:
        st.markdown("**Ficha técnica por prato**")
        pra = listar_pratos()
        ins = listar_insumos()

        if pra.empty:
            st.info("Cadastre ao menos um prato antes.")
        else:
            prato_sel = st.selectbox("Prato", pra["nome"].tolist(), key="fic_prato")

            ficha = ficha_do_prato(prato_sel)
            if ficha.empty:
                st.info(f"'{prato_sel}' ainda não tem ficha técnica.")
            else:
                # Custo total do prato
                merge = ficha.merge(ins, left_on="insumo", right_on="nome", how="left")
                merge["custo"] = merge["quantidade_por_prato"] * merge["custo_unitario"]
                total = merge["custo"].sum()
                st.metric(f"Custo de insumos por '{prato_sel}'", f"R$ {total:.2f}")
                st.dataframe(
                    merge[["insumo", "quantidade_por_prato", "unidade",
                           "custo_unitario", "custo"]].round(3),
                    use_container_width=True,
                )

            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                with st.form("form_add_ficha", clear_on_submit=True):
                    st.markdown("**Adicionar / atualizar item**")
                    insumo_f = st.selectbox("Insumo",
                                             ins["nome"].tolist() if not ins.empty else [])
                    qtd_f = st.number_input("Quantidade por prato",
                                             min_value=0.0, step=0.01, value=None)
                    if st.form_submit_button("Adicionar") and qtd_f is not None:
                        try:
                            adicionar_item_ficha(prato_sel, insumo_f, qtd_f)
                            st.success("Item adicionado.")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            with c2:
                if not ficha.empty:
                    with st.form("form_rm_ficha", clear_on_submit=True):
                        st.markdown("**Remover item**")
                        insumo_rm = st.selectbox("Insumo", ficha["insumo"].tolist(),
                                                  key="rm_ficha_ins")
                        if st.form_submit_button("Remover"):
                            try:
                                remover_item_ficha(prato_sel, insumo_rm)
                                st.success("Item removido.")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))