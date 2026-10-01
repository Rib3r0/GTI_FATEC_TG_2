import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import subprocess
import sys
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from src.config import PROCESSED_DIR, MODELS_DIR
from src.data_loader import carregar_vendas

st.set_page_config(
    page_title="Previsão de Demanda — Restaurante",
    page_icon="🍽️",
    layout="wide",
)

def _reprocessar():
    """Re-executa o pipeline de negócio e limpa o cache do Streamlit."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
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
st.sidebar.title("⚙️ Painel")
st.sidebar.caption("Sistema de previsão de demanda e redução de desperdício")

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

n_falta = int((alertas["status"] == "FALTA").sum())
n_excesso = int((alertas["status"] == "EXCESSO").sum())
capital_risco = float(alertas["custo_excesso"].sum())
custo_compra = float(
    (alertas.loc[alertas["status"] == "FALTA", "saldo"].abs()
     * alertas.loc[alertas["status"] == "FALTA", "custo_unitario"]).sum()
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WAPE (ensemble)", f"{wape_final:.2f}%" if wape_final else "—")
c2.metric("Insumos em falta", n_falta)
c3.metric("Insumos com excesso", n_excesso)
c4.metric("Compra sugerida", f"R$ {custo_compra:,.2f}")
c5.metric("Capital em risco", f"R$ {capital_risco:,.2f}")


# ------------------------------------------------------------------
# ABAS
# ------------------------------------------------------------------
aba_hist, aba_modelo, aba_prev, aba_insumos, aba_alertas, aba_estoque = st.tabs([
    "📊 Histórico",
    "🤖 Modelo",
    "🔮 Previsão",
    "🥕 Insumos",
    "🚨 Alertas",
    "📦 Estoque",
])


# ---- Histórico
with aba_hist:
    st.subheader("Vendas históricas por prato")
    fig = px.line(
        vendas, x="data", y="quantidade", color="prato",
        title="Vendas diárias (últimos 12 meses)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vendas médias por dia da semana")
    vendas["dia_semana"] = vendas["data"].dt.day_name()
    ordem = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    media_dow = (vendas.groupby(["dia_semana", "prato"], observed=True)["quantidade"]
                       .mean().reset_index())
    fig2 = px.bar(media_dow, x="dia_semana", y="quantidade",
                  color="prato", barmode="group",
                  category_orders={"dia_semana": ordem})
    st.plotly_chart(fig2, use_container_width=True)


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
    st.subheader("Previsão dos próximos 7 dias")
    # Reconstrói previsão a partir de data/processed
    from src.predict import prever_proximos_dias
    try:
        prev = prever_proximos_dias(7)
        fig3 = px.line(prev, x="data", y="previsao", color="prato", markers=True,
                       title="Demanda prevista")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(prev.pivot(index="data", columns="prato", values="previsao"),
                     use_container_width=True)
    except Exception as e:
        st.error(f"Não foi possível gerar previsão: {e}")


# ---- Insumos
with aba_insumos:
    st.subheader("Necessidade diária de insumos (kg)")
    if not necessidade_diaria.empty:
        pivot = necessidade_diaria.pivot_table(
            index="data", columns="insumo", values="quantidade_necessaria", aggfunc="sum"
        ).round(2)
        st.dataframe(pivot, use_container_width=True)

        fig4 = px.bar(
            necessidade_diaria, x="data", y="quantidade_necessaria",
            color="insumo", barmode="stack",
            title="Consumo previsto por dia",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ---- Alertas
with aba_alertas:
    st.subheader("Status dos insumos")

    def cor_status(s):
        return {
            "FALTA": "background-color: #E53E3E",
            "EXCESSO": "background-color: #FFB300",
            "OK": "background-color: #87A96B",
        }.get(s, "")

    cols_show = ["insumo", "unidade", "estoque_atual", "necessidade_total",
                 "saldo", "status", "custo_excesso"]
    st.dataframe(
        alertas[cols_show].style.map(cor_status, subset=["status"]),
        use_container_width=True,
    )

    st.subheader("Sugestão de compra")
    falta = alertas[alertas["status"] == "FALTA"].copy()
    if falta.empty:
        st.success("Nenhuma compra necessária para o período.")
    else:
        falta["qtd_sugerida"] = (-falta["saldo"] * 1.10).round(2)
        falta["custo_estimado"] = (falta["qtd_sugerida"] * falta["custo_unitario"]).round(2)
        st.dataframe(
            falta[["insumo", "unidade", "qtd_sugerida", "custo_unitario", "custo_estimado"]],
            use_container_width=True,
        )
        st.info(f"Total estimado: **R$ {falta['custo_estimado'].sum():,.2f}**")

    if not alertas[alertas["status"] == "EXCESSO"].empty:
        st.subheader("⚠️ Insumos com excesso (risco de desperdício)")
        exc = alertas[alertas["status"] == "EXCESSO"]
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
        c1, c2, c3 = st.columns([2, 2, 1])
        f_tipo = c1.multiselect("Tipo", sorted(movs["tipo"].unique().tolist()),
                                default=sorted(movs["tipo"].unique().tolist()))
        f_insumo = c2.multiselect("Insumo", sorted(movs["insumo"].unique().tolist()),
                                   default=sorted(movs["insumo"].unique().tolist()))
        limite = c3.number_input("Máx. registros", min_value=10, max_value=1000,
                                  value=200, step=10)

        filtro = movs[movs["tipo"].isin(f_tipo) & movs["insumo"].isin(f_insumo)]
        st.dataframe(
            filtro.sort_values("data", ascending=False).head(int(limite)),
            use_container_width=True,
        )
        st.caption(f"{len(filtro)} movimentações no filtro atual.")