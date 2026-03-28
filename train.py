"""Train 2D UNet on BraTS H5 slices; saves best checkpoint by validation Dice."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.dataset_loader import get_train_val_dataloaders  # noqa: E402
from models.unet import UNet  # noqa: E402


def dice_multiclass_mean(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Dice over foreground classes 1..C-1 (skip background 0)."""
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(
        0, 3, 1, 2
    ).float()
    dices: list[torch.Tensor] = []
    for c in range(1, num_classes):
        p = probs[:, c].reshape(probs.size(0), -1)
        t = target_oh[:, c].reshape(probs.size(0), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        d = (2 * inter + eps) / (union + eps)
        dices.append(d.mean())
    return torch.stack(dices).mean()


def dice_multilabel_mean(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dices: list[torch.Tensor] = []
    for c in range(probs.shape[1]):
        p = probs[:, c].reshape(probs.size(0), -1)
        t = target[:, c].reshape(target.size(0), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        d = (2 * inter + eps) / (union + eps)
        dices.append(d.mean())
    return torch.stack(dices).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 2D UNet on BraTS H5 slices")
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Folder containing volume_*_slice_*.h5 (default: env BRATS_DATA_ROOT)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--target-h",
        type=int,
        default=128,
        help="Resize H; use 0 together with --target-w 0 for full 240 resolution",
    )
    p.add_argument("--target-w", type=int, default=128)
    p.add_argument(
        "--mask-mode",
        choices=("multiclass", "multilabel"),
        default="multiclass",
    )
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(_ROOT / "checkpoints"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or os.environ.get("BRATS_DATA_ROOT")
    if not data_root:
        raise SystemExit(
            "Set --data-root or environment variable BRATS_DATA_ROOT to your H5 folder."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.target_h <= 0 or args.target_w <= 0:
        if args.target_h != 0 or args.target_w != 0:
            raise SystemExit("Use both --target-h 0 and --target-w 0 to disable resizing.")
        target_hw = None
    else:
        target_hw = (args.target_h, args.target_w)

    train_loader, val_loader = get_train_val_dataloaders(
        data_root=data_root,
        val_frac=args.val_frac,
        seed=args.seed,
        batch_size=args.batch_size,
        target_hw=target_hw,
        mask_mode=args.mask_mode,
        normalize=not args.no_normalize,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if args.mask_mode == "multiclass":
        n_classes = 4
        criterion = nn.CrossEntropyLoss()
    else:
        n_classes = 3
        criterion = nn.BCEWithLogitsLoss()

    model = UNet(n_channels=4, n_classes=n_classes, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss.item())

        train_loss /= max(n_batches, 1)

        model.eval()
        val_dice_sum = 0.0
        val_n = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="val"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(images)
                    if args.mask_mode == "multiclass":
                        val_loss_sum += criterion(logits, targets).item()
                        d = dice_multiclass_mean(logits, targets, n_classes)
                    else:
                        val_loss_sum += criterion(logits, targets).item()
                        d = dice_multilabel_mean(logits, targets)
                val_dice_sum += d.item()
                val_n += 1

        mean_val_dice = val_dice_sum / max(val_n, 1)
        mean_val_loss = val_loss_sum / max(val_n, 1)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={mean_val_loss:.4f} val_dice_mean={mean_val_dice:.4f}"
        )

        if mean_val_dice > best_dice:
            best_dice = mean_val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "val_dice_mean": best_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mask_mode": args.mask_mode,
                    "n_classes": n_classes,
                    "target_hw": target_hw,
                },
                best_path,
            )
            print(f"  saved best checkpoint -> {best_path} (dice={best_dice:.4f})")


if __name__ == "__main__":
    main()
