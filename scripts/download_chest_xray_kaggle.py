#!/usr/bin/env python3
"""
Baixa o dataset 'Chest X-Ray Images (Pneumonia)' via API do Kaggle.

Pré-requisitos:
  pip install kaggle
  ~/.kaggle/kaggle.json com credenciais (ou variáveis KAGGLE_USERNAME / KAGGLE_KEY)

Uso típico (na raiz do projeto):
  python scripts/download_chest_xray_kaggle.py --out data/

O zip descompacta em data/chest-xray-pneumonia/chest_xray/ em muitos casos;
copie ou renomeie para data/chest_xray/ conforme src/config.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET = "paultimothymooney/chest-xray-pneumonia"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Chest X-Ray (Kaggle)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Diretório onde baixar e extrair",
    )
    parser.add_argument("--skip-download", action="store_true", help="Só extrai zip existente")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    zip_path = args.out / "chest-xray-pneumonia.zip"

    if not args.skip_download:
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(args.out)],
                check=True,
            )
        except FileNotFoundError:
            print(
                "Comando 'kaggle' não encontrado. Instale: pip install kaggle",
                file=sys.stderr,
            )
            return 1
        except subprocess.CalledProcessError as e:
            print("Falha no download. Verifique ~/.kaggle/kaggle.json", file=sys.stderr)
            return e.returncode

    if not zip_path.is_file():
        print(f"Arquivo esperado não encontrado: {zip_path}", file=sys.stderr)
        return 2

    extract_to = args.out
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    expected = extract_to / "chest_xray"
    legacy = extract_to / "chest-xray-pneumonia"
    if not expected.is_dir() and legacy.is_dir():
        # alguns zips descompactam com outro nome de pasta raiz
        nested = legacy / "chest_xray"
        if nested.is_dir():
            shutil.move(str(nested), str(expected))
            if not any(legacy.iterdir()):
                legacy.rmdir()
        else:
            shutil.move(str(legacy), str(expected))
        print(f"Pastas alinhadas em: {expected}")
    elif expected.is_dir():
        print(f"Extraído em: {expected}")

    print(
        "Confira se existem train/NORMAL, train/PNEUMONIA, test/... dentro de chest_xray. "
        "Use --data-root se usar outro caminho."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
