"""Caminhos e hiperparâmetros compartilhados."""

from pathlib import Path

# Raiz do repositório (pasta acima de src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset Kaggle "Chest X-Ray Images (Pneumonia)" após extração
# Estrutura esperada: DATA_ROOT/train/{NORMAL,PNEUMONIA}, DATA_ROOT/test/...
def _resolve_default_data_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "chest_xray",
        PROJECT_ROOT / "src" / "data" / "chest_xray",
    ]
    for p in candidates:
        if (p / "train").is_dir():
            return p
    return candidates[0]


DEFAULT_DATA_ROOT = _resolve_default_data_root()

# Transfer learning / entrada ResNet-50
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_FRACTION = 0.15
SEED = 42

# Normalização ImageNet (ResNet pré-treinado)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
