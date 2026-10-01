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
    # Perecíveis — estoque curto (2-4 dias), alguns em falta de propósito
    "alface":       40.0,   # necessidade ~52 kg / 7d → FALTA (perece rápido)
    "tomate":       70.0,   # necessidade ~56 kg / 7d → OK
    "queijo":       20.0,   # necessidade ~46 kg / 7d → FALTA (caro, compra semanal)

    # Secos — estoque de ~1 semana
    "arroz":       180.0,   # necessidade ~165 kg / 7d → OK
    "feijao":       90.0,   # necessidade ~94 kg / 7d → leve FALTA
    "farinha":      30.0,   # necessidade ~12 kg / 7d → EXCESSO
    "cebola":       50.0,   # necessidade ~33 kg / 7d → OK
    
    # Raízes / acompanhamentos
    "batata":       70.0,   # necessidade ~95 kg / 7d → leve FALTA

    # Proteínas — estoque de 5-6 dias
    "carne_bovina":150.0,   # necessidade ~172 kg / 7d → leve FALTA
    "frango":      200.0,   # necessidade ~224 kg / 7d → FALTA
}