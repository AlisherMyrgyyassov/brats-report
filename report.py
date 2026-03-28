"""Run 2D UNet on one BraTS H5 volume (slice stack), extract mask features, optional DeepSeek report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.dataset_loader import _parse_h5_path  # noqa: E402
from models.unet import UNet  # noqa: E402


def gather_volume_paths(data_root: Path, volume_id: int) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    paths: list[tuple[int, Path]] = []
    for p in data_root.glob(f"volume_{volume_id}_slice_*.h5"):
        parsed = _parse_h5_path(p)
        if parsed is None:
            continue
        pid, sl = parsed
        if pid != volume_id:
            continue
        paths.append((sl, p))
    if not paths:
        raise FileNotFoundError(
            f"No slices matching volume_{volume_id}_slice_*.h5 under {data_root}"
        )
    paths.sort(key=lambda x: x[0])
    return [p for _, p in paths]


def load_h5_image(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "image" not in f:
            raise KeyError(f"{path}: missing 'image'")
        image = np.asarray(f["image"][...], dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError(f"{path}: image shape {image.shape}, expected (H,W,4)")
    return image


def normalize_image_chw(image_hw4: np.ndarray) -> torch.Tensor:
    """Match BraTSH5SliceDataset: per-channel mean/std on nonzeros."""
    image = torch.from_numpy(image_hw4).permute(2, 0, 1).contiguous()
    for c in range(image.shape[0]):
        ch = image[c]
        nz = ch != 0
        v = ch[nz]
        if v.numel() > 0:
            mean = v.mean()
            std = v.std().clamp_min(1e-6)
            out = (ch - mean) / std
            out = torch.nan_to_num(out, nan=0.0)
            out[~nz] = 0.0
            image[c] = out
    return image


def load_checkpoint_model(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[UNet, dict]:
    ckpt_path = Path(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    mask_mode = ckpt.get("mask_mode", "multiclass")
    n_classes = int(ckpt.get("n_classes", 4 if mask_mode == "multiclass" else 3))
    target_hw = ckpt.get("target_hw", None)
    if target_hw is not None:
        target_hw = tuple(int(x) for x in target_hw)

    model = UNet(n_channels=4, n_classes=n_classes, bilinear=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "mask_mode": mask_mode,
        "n_classes": n_classes,
        "target_hw": target_hw,
        "epoch": ckpt.get("epoch"),
        "val_dice_mean": ckpt.get("val_dice_mean"),
    }
    return model, meta


def resize_pred_to_hw(pred: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """pred: (H0,W0) long -> (H,W) long nearest."""
    if pred.shape[0] == h and pred.shape[1] == w:
        return pred
    x = pred.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(h, w), mode="nearest")
    return x.squeeze(0).squeeze(0).long()


def logits_to_label_map_multilabel(
    logits: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """(1,3,h,w) -> (h,w) int in 0..3; overlaps resolved ET > ED > NCR."""
    probs = torch.sigmoid(logits.squeeze(0))
    ncr = probs[0] >= threshold
    ed = probs[1] >= threshold
    et = probs[2] >= threshold
    out = torch.zeros(probs.shape[1:], dtype=torch.long, device=logits.device)
    out[ncr] = 1
    out[ed] = 2
    out[et] = 3
    return out


def infer_volume_labels(
    slice_paths: list[Path],
    model: UNet,
    mask_mode: str,
    target_hw: tuple[int, int] | None,
    device: torch.device,
    multilabel_threshold: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Returns:
        pred_dhw: (D,H,W) int64 labels 0..3
        images_dhw4: (D,H,W,4) float32 original (for visualization)
        spatial_shape: (D,H,W) of pred
    """
    preds: list[torch.Tensor] = []
    images: list[np.ndarray] = []

    with torch.no_grad():
        for path in slice_paths:
            img_hw4 = load_h5_image(path)
            images.append(img_hw4)
            chw = normalize_image_chw(img_hw4).to(device)
            h0, w0 = chw.shape[1], chw.shape[2]
            if target_hw is not None:
                th, tw = target_hw
                x = chw.unsqueeze(0)
                x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
            else:
                x = chw.unsqueeze(0)

            logits = model(x)
            if mask_mode == "multiclass":
                pred = logits.argmax(dim=1).squeeze(0)
            else:
                pred = logits_to_label_map_multilabel(logits, multilabel_threshold)

            pred = resize_pred_to_hw(pred.cpu(), h0, w0)
            preds.append(pred)

    pred_stack = torch.stack(preds, dim=0).numpy().astype(np.int64)
    img_stack = np.stack(images, axis=0)
    d, h, w = pred_stack.shape
    return pred_stack, img_stack, (d, h, w)


def exclusive_label_volumes(pred_dhw: np.ndarray) -> tuple[int, int, int, int]:
    """Whole tumor and per-class voxel counts (exclusive ET>ED>NCR)."""
    p = pred_dhw.copy()
    et = p == 3
    ed = (p == 2) & ~et
    ncr = (p == 1) & ~ed & ~et
    n_et = int(et.sum())
    n_ed = int(ed.sum())
    n_ncr = int(ncr.sum())
    n_whole = n_et + n_ed + n_ncr
    return n_whole, n_ncr, n_ed, n_et


def bbox_extent_cm(pred_dhw: np.ndarray, spacing_mm: tuple[float, float, float]) -> list[float]:
    """Axis-aligned bbox over whole tumor; extent per axis in cm."""
    tumor = pred_dhw > 0
    if not tumor.any():
        return [0.0, 0.0, 0.0]
    coords = np.argwhere(tumor)
    d_min, d_max = coords[:, 0].min(), coords[:, 0].max()
    h_min, h_max = coords[:, 1].min(), coords[:, 1].max()
    w_min, w_max = coords[:, 2].min(), coords[:, 2].max()
    ext_d = (d_max - d_min + 1) * spacing_mm[0] / 10.0
    ext_h = (h_max - h_min + 1) * spacing_mm[1] / 10.0
    ext_w = (w_max - w_min + 1) * spacing_mm[2] / 10.0
    return [round(ext_d, 2), round(ext_h, 2), round(ext_w, 2)]


def count_et_lesions(pred_dhw: np.ndarray, min_voxels: int) -> int:
    et = pred_dhw == 3
    if not et.any():
        return 0
    labeled, n = ndimage.label(et)
    count = 0
    for lab in range(1, n + 1):
        if int((labeled == lab).sum()) >= min_voxels:
            count += 1
    return count


def slice_with_max_tumor(pred_dhw: np.ndarray) -> int:
    areas = [(pred_dhw[z] > 0).sum() for z in range(pred_dhw.shape[0])]
    return int(np.argmax(areas))


def save_overlay_png(
    image_hw4: np.ndarray,
    pred_hw: np.ndarray,
    out_path: Path,
    bg_channel: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if bg_channel < 0 or bg_channel >= 4:
        bg_channel = 0
    bg = image_hw4[:, :, bg_channel].astype(np.float32)
    nz = bg != 0
    if nz.any():
        v = bg[nz]
        bg = (bg - v.min()) / (v.max() - v.min() + 1e-8)
    else:
        bg = np.zeros_like(bg)
    bg = np.clip(bg, 0, 1)

    rgba = np.zeros((*bg.shape, 4), dtype=np.float32)
    rgba[..., 0] = bg
    rgba[..., 1] = bg
    rgba[..., 2] = bg
    rgba[..., 3] = 1.0

    colors = {
        1: (0.2, 0.6, 1.0, 0.45),
        2: (1.0, 0.85, 0.2, 0.45),
        3: (1.0, 0.2, 0.2, 0.5),
    }
    for lab, c in colors.items():
        m = pred_hw == lab
        if not m.any():
            continue
        alpha = c[3]
        for k in range(3):
            rgba[..., k] = np.where(m, (1 - alpha) * rgba[..., k] + alpha * c[k], rgba[..., k])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgba)
    ax.set_title("Prediction overlay (NCR/NET=blue, ED=yellow, ET=red)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_legend_png(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.add_patch(plt.Rectangle((0.05, 0.35), 0.08, 0.3, color=(0.2, 0.6, 1.0)))
    ax.text(0.16, 0.5, "NCR/NET", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((0.42, 0.35), 0.08, 0.3, color=(1.0, 0.85, 0.2)))
    ax.text(0.53, 0.5, "ED", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((0.68, 0.35), 0.08, 0.3, color=(1.0, 0.2, 0.2)))
    ax.text(0.79, 0.5, "ET", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_feature_payload(
    pred_dhw: np.ndarray,
    spacing_mm: tuple[float, float, float],
    min_component_voxels: int,
    viz_names: list[str],
) -> dict:
    n_whole, n_ncr, n_ed, n_et = exclusive_label_volumes(pred_dhw)
    if n_whole == 0:
        p_ncr = p_ed = p_et = 0
    else:
        p_ncr = int(round(100.0 * n_ncr / n_whole))
        p_ed = int(round(100.0 * n_ed / n_whole))
        p_et = int(round(100.0 * n_et / n_whole))

    bbox_cm = bbox_extent_cm(pred_dhw, spacing_mm)
    n_lesions = count_et_lesions(pred_dhw, min_component_voxels)

    computed = {
        "tumor_type": "Glioma",
        "bbox_extent_cm_D_H_W": bbox_cm,
        "bbox_axis_note": (
            "Extents follow stacked slice index (D), image row (H), image column (W); "
            "not radiological AP/TV/CC."
        ),
        "num_lesions_et": n_lesions,
        "percent_ncr_net": p_ncr,
        "percent_ed": p_ed,
        "percent_et": p_et,
        "total_tumor_voxels": n_whole,
        "voxel_spacing_mm": list(spacing_mm),
    }

    not_computed = {
        "survival_days": None,
        "midline_shift_mm": None,
        "midline_shift_direction": None,
        "ventricle_level": None,
        "edema_crosses_midline": None,
        "asymmetric_ventricles": None,
        "enlarged_ventricles": None,
        "anatomic_location": None,
        "ventricles_invaded": None,
        "proportion_enhancing_percent": None,
        "deep_wm_invaded": None,
        "multifocal": None,
    }

    return {
        "computed": computed,
        "not_computed": not_computed,
        "visualization_paths": viz_names,
    }


def call_deepseek_report(
    prompt_text: str,
    feature_payload: dict,
    api_key: str,
    model: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    user_content = (
        "Feature JSON (ground truth for what you may state numerically):\n```json\n"
        + json.dumps(feature_payload, indent=2)
        + "\n```"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    choice = resp.choices[0]
    return (choice.message.content or "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segment one H5 volume with trained UNet and generate a mask-based report."
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Folder with volume_*_slice_*.h5 (default: env BRATS_DATA_ROOT)",
    )
    p.add_argument("--volume-id", type=int, default=1, help="Patient / volume id in filenames")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(_ROOT / "checkpoints" / "best.pt"),
    )
    p.add_argument("--out-dir", type=str, default=str(_ROOT / "examples"))
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or omit for auto",
    )
    p.add_argument("--no-llm", action="store_true", help="Skip DeepSeek; save features JSON only")
    p.add_argument(
        "--bg-channel",
        type=int,
        default=3,
        help="MRI channel index 0..3 for overlay background (default 3)",
    )
    p.add_argument("--multilabel-threshold", type=float, default=0.5)
    p.add_argument("--min-component-voxels", type=int, default=50)
    p.add_argument("--deepseek-model", type=str, default="deepseek-chat")
    p.add_argument(
        "--spacing-mm",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("D", "H", "W"),
        help="Voxel spacing in mm per axis (default 1 1 1)",
    )
    return p.parse_args()


def main() -> None:
    """
    Default run: uses BRATS_DATA_ROOT, volume_id=1, checkpoints/best.pt, writes under examples/.
    Set DEEPSEEK_API_KEY in .env for LLM narrative (or use --no-llm).
    """
    load_dotenv(_ROOT / ".env")
    args = parse_args()
    data_root = args.data_root or os.environ.get("BRATS_DATA_ROOT")
    if not data_root:
        raise SystemExit(
            "Set --data-root or environment variable BRATS_DATA_ROOT to your H5 folder."
        )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_paths = gather_volume_paths(Path(data_root), args.volume_id)
    model, meta = load_checkpoint_model(ckpt_path, device)

    pred_dhw, images_dhw4, _ = infer_volume_labels(
        slice_paths,
        model,
        mask_mode=meta["mask_mode"],
        target_hw=meta["target_hw"],
        device=device,
        multilabel_threshold=args.multilabel_threshold,
    )

    z_show = slice_with_max_tumor(pred_dhw)
    overlay_name = f"volume_{args.volume_id}_slice{z_show}_overlay.png"
    legend_name = "segmentation_legend.png"
    overlay_path = out_dir / overlay_name
    legend_path = out_dir / legend_name

    save_overlay_png(
        images_dhw4[z_show],
        pred_dhw[z_show],
        overlay_path,
        bg_channel=args.bg_channel,
    )
    save_legend_png(legend_path)

    spacing = (float(args.spacing_mm[0]), float(args.spacing_mm[1]), float(args.spacing_mm[2]))
    viz_rel = [overlay_name, legend_name]
    payload = build_feature_payload(
        pred_dhw,
        spacing_mm=spacing,
        min_component_voxels=args.min_component_voxels,
        viz_names=viz_rel,
    )

    features_path = out_dir / f"volume_{args.volume_id}_features.json"
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote features -> {features_path}")
    print(f"Wrote visualizations -> {overlay_path}, {legend_path}")

    report_path = out_dir / f"volume_{args.volume_id}_report.md"
    if args.no_llm:
        report_path.write_text(
            "_LLM skipped (--no-llm). See features JSON and PNGs._\n",
            encoding="utf-8",
        )
        print(f"Wrote stub report -> {report_path}")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit(
            "DEEPSEEK_API_KEY missing in environment/.env, or pass --no-llm to skip the API call."
        )

    prompt_path = _ROOT / "promts" / "report_prompt.txt"
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    report_md = call_deepseek_report(
        prompt_text,
        payload,
        api_key=api_key,
        model=args.deepseek_model,
    )
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
