#!/usr/bin/env python3
"""Grad-CAM++ lado a lado (ou dois arquivos) para dois checkpoints — ex.: head_only vs full_finetune."""

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
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, 2)
    m.load_state_dict(ck["model_state"])
    label = ck.get("strategy", weights_path.stem)
    return m.to(device).eval(), str(label)


def run_cam(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    target_class: int,
) -> np.ndarray:
    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    return grayscale_cam


def concat_horizontal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] != h:
        scale = h / a.shape[0]
        nw = int(a.shape[1] * scale)
        a = np.array(Image.fromarray(a).resize((nw, h)))
    if b.shape[0] != h:
        scale = h / b.shape[0]
        nw = int(b.shape[1] * scale)
        b = np.array(Image.fromarray(b).resize((nw, h)))
    return np.concatenate([a, b], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True, help="ex.: checkpoints/resnet50_head_only.pt")
    parser.add_argument("--checkpoint-b", type=Path, required=True, help="ex.: checkpoints/resnet50_full_finetune.pt")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target-class", type=int, default=1, help="0=NORMAL, 1=PNEUMONIA (mapa para essa classe)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "gradcam_compare",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Gera também uma imagem única A|B (lado a lado).",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        print("--image inválido.", file=sys.stderr)
        return 2
    for ck in (args.checkpoint_a, args.checkpoint_b):
        if not ck.is_file():
            print(f"Checkpoint não encontrado: {ck}", file=sys.stderr)
            return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    rgb = Image.open(args.image).convert("RGB")
    tensor = tf(rgb).unsqueeze(0).to(device)
    rgb_np = np.array(rgb.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    models_out: list[tuple[str, nn.Module]] = []
    ma, la = load_model(args.checkpoint_a, device)
    mb, lb = load_model(args.checkpoint_b, device)
    models_out.append((la, ma))
    models_out.append((lb, mb))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_list: list[np.ndarray] = []
    stem = args.image.stem

    for label, m in models_out:
        g = run_cam(m, tensor, device, args.target_class)
        vis = show_cam_on_image(rgb_np, g, use_rgb=True)
        vis_list.append(vis)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        out_path = args.out_dir / f"{stem}_gradcam++_{safe}.png"
        Image.fromarray(vis).save(out_path)
        print(f"Salvo: {out_path}")

    if args.combined and len(vis_list) == 2:
        comb = concat_horizontal(vis_list[0], vis_list[1])
        combined_path = args.out_dir / f"{stem}_gradcam++_A_vs_B.png"
        Image.fromarray(comb).save(combined_path)
        print(f"Combinado: {combined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
