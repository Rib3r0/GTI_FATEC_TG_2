# Previsão de Demanda e Redução de Desperdício em Restaurantes

Sistema de gestão de estoque com machine learning para prever demanda,
traduzir previsão em necessidade de insumos e gerar alertas de falta,
excesso e desperdício.

Projeto de Trabalho de Graduação — Machine Learning aplicado à operação
de restaurantes.

---

## 🎯 O problema

Restaurantes operam com margens apertadas e enfrentam duas dores opostas:

- **Ruptura**: falta de insumo em dia de pico → perda de venda.
- **Desperdício**: excesso de compra ou preparo → custo afundado.

A gestão típica é reativa: o gestor decide com base em intuição e no
"olhômetro". Este projeto propõe um sistema que **prevê demanda,
traduz em insumos e recomenda ações** de compra e produção.

## 💡 Funcionalidades

- **Previsão de demanda** — 4 modelos + 2 ensembles, escolha automática
  pelo menor WAPE.
- **Conversão em insumos** — ficha técnica liga prato vendido a consumo
  de insumos.
- **Controle de estoque** — livro-razão de movimentações: entradas,
  saídas por venda, ajustes de auditoria e perdas.
- **Alertas** — classificação de cada insumo em FALTA / OK / EXCESSO,
  com sugestão de compra e capital em risco.
- **Análise de desperdício** — taxa, motivos, top insumos, balanço
  contábil e consumo teórico vs real.
- **Dashboard interativo** — filtros globais por período, prato e insumo.

## 🏗️ Arquitetura

```text
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  data_generator  │ →  │      train       │ →  │     predict      │
│  dados sintéticos│    │  4 modelos + 2   │    │  previsão 7d     │
│  vendas/insumos  │    │    ensembles     │    │    recursiva     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│    inventory     │ ←  │     business     │ ←  │  ficha_técnica   │
│  movimentações   │    │  necessidade →   │    │  consumo por     │
│  saldo atual     │    │    insumos       │    │     prato        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│      waste       │    │      alerts      │
│  perdas, balanço │    │  FALTA/OK/EXCESSO│
└──────────────────┘    └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌──────────────┐
              │   app.py     │
              │  Streamlit   │
              └──────────────┘
```

## 📁 Estrutura de pastas

```text
restaurant-demand-forecast/
├── app.py                          # Dashboard Streamlit
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                        # vendas, insumos, ficha, movimentações
│   └── processed/                  # alertas, comparações, análises
├── models/
│   └── modelos_demanda.pkl         # modelos treinados
└── src/
    ├── config.py                   # caminhos e parâmetros globais
    ├── data_generator.py           # gera dados sintéticos
    ├── data_loader.py              # carrega CSVs
    ├── features.py                 # engenharia de features
    ├── train.py                    # treina e compara modelos
    ├── predict.py                  # previsão recursiva 7d
    ├── evaluate.py                 # MAE, RMSE, MAPE, WAPE, viés
    ├── inventory.py                # livro-razão de movimentações
    ├── business.py                 # previsão → necessidade → alertas
    ├── alerts.py                   # classificação e sugestão de compra
    └── waste.py                    # análise de desperdício
```

## 🚀 Instalação

Requer Python 3.10+.

```bash
git clone <seu-repo>
cd restaurant-demand-forecast

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## ▶️ Como usar

### 1. Rodar o pipeline pela primeira vez

```bash
python -m src.data_generator       # gera dados sintéticos
python -m src.train                # treina 4 modelos + 2 ensembles
python -m src.inventory            # inicializa o estoque (livro-razão)
python -m src.business             # previsão + alertas + desperdício
```

### 2. Abrir o dashboard

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`.

### 3. Alternativa: rodar tudo pela interface

No dashboard, clique em **🔄 Rodar pipeline completo** na lateral.
O sistema executa todos os passos em sequência.

## 🧠 Modelos avaliados

Validação temporal — últimos 30 dias do histórico.

| Modelo | WAPE (%) | MAE |
|---|---|---|
| Baseline (média móvel 7d) | 19,64 | 8,84 |
| Ridge | 13,03 | 5,87 |
| LightGBM | 12,64 | 5,69 |
| Random Forest | 12,01 | 5,41 |
| Ensemble (média simples) | 11,69 | 5,26 |
| **Ensemble (média ponderada)** | **11,69** | **5,26** |

O **ensemble ponderado** foi escolhido como modelo final por menor WAPE,
menor MAE e maior robustez.

## 📊 Métricas

- **MAE** — erro absoluto médio (unidades).
- **RMSE** — raiz do erro quadrático médio.
- **MAPE** — erro percentual médio (sensível a valores baixos).
- **WAPE** — erro absoluto ponderado (métrica principal).
- **Viés** — média dos erros com sinal (detecta sub/superestimação).

## 🛠️ Tecnologias

- **Python 3.12**
- **pandas**, **numpy** — manipulação de dados
- **scikit-learn**, **lightgbm** — modelagem
- **Streamlit** — dashboard
- **Plotly** — visualizações
- **joblib** — persistência de modelos

## 📈 Resultados principais

- Redução de **~40% no WAPE** vs. baseline.
- Identificação proativa de insumos em risco de ruptura.
- Quantificação em R$ do **capital em risco de desperdício**.
- Tradução automática de previsão de pratos em necessidade de insumos.

## 🔬 Limitações

- Dados sintéticos (dados reais de restaurante são proprietários).
- Horizonte de previsão de 7 dias (previsões longas degradam).
- Ficha técnica fixa (em produção, variaria com sazonalidade e
  fornecedor).
- Não considera lead time variável de fornecedores.

## 🚧 Trabalhos futuros

- Integração com POS/PDV real.
- Classificador de risco de desperdício (segunda frente de ML).
- Otimização de compras com programação linear.
- App mobile para registro em tempo real pela cozinha.

## 📄 Licença

Projeto acadêmico — uso livre para fins educacionais.