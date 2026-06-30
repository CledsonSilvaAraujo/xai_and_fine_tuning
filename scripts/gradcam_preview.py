#!/usr/bin/env python3
"""Um mapa Grad-CAM++ para uma imagem — requer checkpoint do train_baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def load_model(weights_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 2)
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "gradcam_preview.png")
    parser.add_argument("--target-class", type=int, default=1, help="0=NORMAL, 1=PNEUMONIA")
    args = parser.parse_args()

    if not args.checkpoint.is_file() or not args.image.is_file():
        print("checkpoint ou imagem inválidos.", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    rgb = Image.open(args.image).convert("RGB")
    tensor = tf(rgb).unsqueeze(0).to(device)

    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(args.target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    rgb_np = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(args.out)
    print(f"Salvo: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
