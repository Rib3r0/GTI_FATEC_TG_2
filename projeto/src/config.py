import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Parâmetros do negócio
HORIZONTE_PREVISAO = 7       # dias à frente
DIAS_TREINO = 365            # histórico sintético
SEED = 42

# Estoque inicial de exemplo (unidades)
ESTOQUE_INICIAL = {
    "arroz": 20.0,
    "feijao": 15.0,
    "carne_bovina": 30.0,
    "frango": 25.0,
    "batata": 40.0,
    "tomate": 20.0,
    "cebola": 20.0,
    "alface": 10.0,
    "queijo": 8.0,
    "farinha": 15.0,
}