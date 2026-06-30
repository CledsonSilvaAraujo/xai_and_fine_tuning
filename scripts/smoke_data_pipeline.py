#!/usr/bin/env python3
"""Carrega uma batch do Chest X-Ray se data/chest_xray existir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_ROOT
from src.data import ChestXRayBinaryDataModule, class_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Pasta com train/ e test/ (subpastas NORMAL, PNEUMONIA)",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(
            "Dataset não encontrado.\n"
            "1) Baixe no Kaggle: 'Chest X-Ray Images (Pneumonia)'.\n"
            "2) Extraia para data/chest_xray/ com estrutura:\n"
            "   chest_xray/train/NORMAL, chest_xray/train/PNEUMONIA,\n"
            "   chest_xray/test/NORMAL, chest_xray/test/PNEUMONIA\n"
            "Opcional (CLI Kaggle): kaggle datasets download -d paultimothymooney/"
            "chest-xray-pneumonia -p data --unzip\n"
            "(ajuste o caminho final para data/chest_xray se o zip criar outro nome.)",
            file=sys.stderr,
        )
        return 2

    dm = ChestXRayBinaryDataModule(args.data_root)
    print(dm.describe())
    print("Classes:", class_names())

    train = dm.train_loader()
    x, y = next(iter(train))
    print("batch x:", tuple(x.shape), "dtype:", x.dtype)
    print("batch y:", y[:8], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
