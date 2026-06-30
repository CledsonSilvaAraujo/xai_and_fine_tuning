#!/usr/bin/env python3
"""Confere PyTorch, torchvision e pytorch_grad_cam ( Grad-CAM / Grad-CAM++ )."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    errors: list[str] = []
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        print(f"PyTorch {torch.__version__}")
        print(f"torchvision {torchvision.__version__}")
        print(f"CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(str(e))

    try:
        from pytorch_grad_cam import GradCAM  # noqa: F401

        print("pytorch_grad_cam import OK (pacote pip: grad-cam)")
    except ImportError as e:
        errors.append(f"Grad-CAM: {e}")

    try:
        from pytorch_grad_cam import GradCAMPlusPlus  # noqa: F401

        print("GradCAMPlusPlus disponível")
    except ImportError as e:
        errors.append(f"GradCAM++: {e}")

    if errors:
        print("Problemas:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
