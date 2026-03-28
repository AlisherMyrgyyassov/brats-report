"""BraTS 2020-style H5 slice dataset: flat folder of volume_{id}_slice_{k}.h5 files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

_FILENAME_RE = re.compile(
    r"^volume_(?P<pid>\d+)_slice_(?P<slice>\d+)\.h5$", re.IGNORECASE
)

MaskMode = Literal["multiclass", "multilabel"]


def _parse_h5_path(path: Path) -> tuple[int, int] | None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group("pid")), int(m.group("slice"))


def _gather_h5_files(data_root: Path) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is not a directory: {data_root}")
    paths = sorted(p for p in data_root.glob("*.h5") if _parse_h5_path(p) is not None)
    if not paths:
        raise FileNotFoundError(f"No volume_*_slice_*.h5 files under {data_root}")
    return paths


def split_by_patient(
    paths: list[Path],
    val_frac: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Assign all slices of a patient to either train or val (no leakage)."""
    rng = np.random.default_rng(seed)
    by_patient: dict[int, list[Path]] = defaultdict(list)
    for p in paths:
        parsed = _parse_h5_path(p)
        assert parsed is not None
        pid, _ = parsed
        by_patient[pid].append(p)

    pids = sorted(by_patient.keys())
    if len(pids) < 2:
        if len(paths) == 1:
            return list(paths), list(paths)
        idx = np.arange(len(paths))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(paths) * val_frac)))
        n_val = min(n_val, len(paths) - 1)
        val_idx = idx[-n_val:]
        train_idx = idx[:-n_val]
        val_paths = [paths[i] for i in val_idx]
        train_paths = [paths[i] for i in train_idx]
        return train_paths, val_paths

    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_frac)))
    val_pids = set(pids[-n_val:])

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for pid in pids:
        bucket = val_paths if pid in val_pids else train_paths
        bucket.extend(sorted(by_patient[pid], key=lambda x: _parse_h5_path(x) or (0, 0)))

    return train_paths, val_paths


def mask_hw3_to_multiclass_label(mask_hw3: np.ndarray) -> np.ndarray:
    """(H,W,3) one-hot-style -> (H,W) int64 in {0,1,2,3}; 0 = background."""
    if mask_hw3.dtype != np.float32 and mask_hw3.dtype != np.float64:
        mask_hw3 = mask_hw3.astype(np.float32)
    active = mask_hw3.sum(axis=-1) > 0.5
    label = np.zeros(mask_hw3.shape[:2], dtype=np.int64)
    idx = np.argmax(mask_hw3, axis=-1).astype(np.int64)
    label[active] = idx[active] + 1
    return label


class BraTSH5SliceDataset(Dataset):
    """
    Reads `image` (H,W,4) and `mask` (H,W,3) from each HDF5 file.

    - multiclass: target is (H,W) long with values 0..3 (bg + 3 foreground channels as one-hot).
    - multilabel: target is float (3,H,W) in [0,1] for BCEWithLogitsLoss (overlapping masks OK).
    """

    def __init__(
        self,
        paths: list[Path],
        target_hw: tuple[int, int] | None = None,
        mask_mode: MaskMode = "multiclass",
        normalize: bool = True,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.target_hw = target_hw
        self.mask_mode = mask_mode
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def _load_arrays(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(path, "r") as f:
            if "image" not in f or "mask" not in f:
                raise KeyError(
                    f"{path}: expected datasets 'image' and 'mask', got {list(f.keys())}"
                )
            image = np.asarray(f["image"][...], dtype=np.float32)
            mask = np.asarray(f["mask"][...], dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"{path}: image shape {image.shape}, expected (H,W,4)")
        if mask.ndim != 3 or mask.shape[-1] != 3:
            raise ValueError(f"{path}: mask shape {mask.shape}, expected (H,W,3)")
        return image, mask

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path = self.paths[index]
        image_hw4, mask_hw3 = self._load_arrays(path)

        # (H,W,C) -> (C,H,W)
        image = torch.from_numpy(image_hw4).permute(2, 0, 1).contiguous()
        if self.normalize:
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

        if self.mask_mode == "multiclass":
            label_np = mask_hw3_to_multiclass_label(mask_hw3)
            target = torch.from_numpy(label_np).long()
        else:
            target = torch.from_numpy(mask_hw3).permute(2, 0, 1).contiguous().clamp(0.0, 1.0)

        if self.target_hw is not None:
            th, tw = self.target_hw
            image = image.unsqueeze(0)
            image = F.interpolate(image, size=(th, tw), mode="bilinear", align_corners=False)
            image = image.squeeze(0)
            if self.mask_mode == "multiclass":
                target = target.unsqueeze(0).unsqueeze(0).float()
                target = F.interpolate(target, size=(th, tw), mode="nearest")
                target = target.squeeze(0).squeeze(0).long()
            else:
                target = target.unsqueeze(0)
                target = F.interpolate(target, size=(th, tw), mode="nearest")
                target = target.squeeze(0)

        return image, target


def get_train_val_dataloaders(
    data_root: str | Path,
    val_frac: float = 0.2,
    seed: int = 42,
    batch_size: int = 4,
    target_hw: tuple[int, int] | None = None,
    mask_mode: MaskMode = "multiclass",
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    paths = _gather_h5_files(Path(data_root))
    train_paths, val_paths = split_by_patient(paths, val_frac=val_frac, seed=seed)
    if not train_paths:
        raise RuntimeError("Train split is empty; lower val_frac or check data.")
    if not val_paths:
        raise RuntimeError("Val split is empty; need at least 2 patients or lower val_frac.")

    train_ds = BraTSH5SliceDataset(
        train_paths,
        target_hw=target_hw,
        mask_mode=mask_mode,
        normalize=normalize,
    )
    val_ds = BraTSH5SliceDataset(
        val_paths,
        target_hw=target_hw,
        mask_mode=mask_mode,
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    return train_loader, val_loader
