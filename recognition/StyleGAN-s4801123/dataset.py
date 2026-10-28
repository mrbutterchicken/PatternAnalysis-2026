"""
Contains the data loader for loading and preprocessing your data, 
including leakage-free train/validation/test splitting functions 
and data augmentations.
"""
import copy, csv, json, math, os, random, time, zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "Runtime > Change runtime type > A100 GPU"
DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True     # A100: fast TF32 matmuls/convs, no mixed-precision headaches
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
print(torch.cuda.get_device_name(0))

# ---- paths ----------------------------------------------------------------------------------
ZIP_SOURCE = "/content/drive/MyDrive/"        # folder holding the zip (or the .zip file itself)
UNZIP_TO   = "/content"
OUT_DIR    = "/content/drive/MyDrive/stylegan-sagittal"   # checkpoints, samples, logs (on Drive: survives disconnects)

# ---- data -----------------------------------------------------------------------------------
WEIGHTINGS = ["T1"]    # sub-folders of the zip to train on; None = pool all weightings (mixed contrasts)
VAL_FRAC   = 0.10      # fraction of SUBJECTS held out of training (only used by the PREDICT audits)
SEED       = 0

# ---- training -------------------------------------------------------------------------------
BATCH          = 32

# =============================================================================
# DATASET   unzip -> subject-level split -> all slices into RAM (uint8) -> fast GPU batches
# =============================================================================
def find_zip(src):
    src = Path(src)
    if src.is_file():
        return src
    zips = sorted(src.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError(f"No .zip found in {src}  (on Colab the folder is 'MyDrive', capital M and D)")
    if len(zips) > 1:
        print("Several zips found, using the newest:", zips[-1].name)
    return zips[-1]


def unzip_pngs(zip_path, dest):
    """Extract only the PNGs (skips macOS __MACOSX/._* junk). Returns their paths."""
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist()
                   if n.lower().endswith(".png") and "__MACOSX" not in n and not Path(n).name.startswith("._")]
        todo = [n for n in members if not (Path(dest) / n).exists()]
        if todo:
            print(f"Extracting {len(todo)} PNGs to {dest} ...")
            z.extractall(dest, todo)
    return [Path(dest) / n for n in members]


def subject_of(path):
    """Converter filenames look like <subject>_<series>_<slice>.png"""
    return Path(path).name.split("_")[0]


def list_files():
    files = unzip_pngs(find_zip(ZIP_SOURCE), UNZIP_TO)
    counts = defaultdict(int)
    for f in files:
        counts[f.parent.name] += 1
    print("Slices per folder in the zip:", dict(counts))
    if WEIGHTINGS:
        files = [f for f in files if f.parent.name in WEIGHTINGS]
    if not files:
        raise RuntimeError(f"No PNGs found in folders {WEIGHTINGS}. Folders in the zip: {list(counts)}")
    return sorted(files)


def split_by_subject(files, val_frac, seed):
    """Hold out whole subjects, so slices of one person never sit in both sets."""
    subjects = sorted({subject_of(f) for f in files})
    random.Random(seed).shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_frac)) if len(subjects) > 1 and val_frac > 0 else 0
    held = set(subjects[:n_val])
    train = [f for f in files if subject_of(f) not in held]
    val = [f for f in files if subject_of(f) in held]
    print(f"{len(subjects)} subjects -> {len(subjects) - n_val} train / {n_val} held out "
          f"({len(train)} train slices, {len(val)} held-out slices)")
    return train, val


class MRISlices(torch.utils.data.Dataset):
    """Every slice held in RAM as uint8 [N,1,H,W]; items come out as float [1,H,W] in [-1,1]."""
    def __init__(self, files):
        self.files = list(files)
        imgs = [np.asarray(Image.open(f).convert("L")) for f in tqdm(self.files, desc="Loading PNGs")]
        shapes = {im.shape for im in imgs}
        assert len(shapes) == 1, f"PNG sizes differ: {shapes}"
        h, w = imgs[0].shape
        assert h == w and h & (h - 1) == 0, f"need square power-of-two images, got {h}x{w}"
        self.data = torch.from_numpy(np.stack(imgs))[:, None].contiguous()
        self.res = h
        self.subjects = [subject_of(f) for f in self.files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i].float() / 127.5 - 1


def batches(ds, batch_size):
    """Endless stream of shuffled GPU batches in [-1,1] (the whole set lives on the GPU as uint8)."""
    data = ds.data.to(DEVICE)
    n = len(data)
    while True:
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - batch_size + 1, batch_size):
            yield data[perm[i:i + batch_size]].float() / 127.5 - 1


def show(imgs, nrow=8, title=None, size=1.6):
    """imgs: [N,1,H,W] in [-1,1]"""
    grid = make_grid((imgs.float().cpu().clamp(-1, 1) + 1) / 2, nrow=nrow, padding=2)
    plt.figure(figsize=(nrow * size, math.ceil(len(imgs) / nrow) * size))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def prepare_data():
    files = list_files()
    train_files, val_files = split_by_subject(files, VAL_FRAC, SEED)
    train = MRISlices(train_files)
    val = MRISlices(val_files) if val_files else None
    return train, val


train_ds, val_ds = prepare_data()
RES = train_ds.res
assert len(train_ds) >= BATCH, f"only {len(train_ds)} training slices, fewer than BATCH={BATCH}"
print(f"{len(train_ds)} training slices at {RES}x{RES}" + (f" | {len(val_ds)} held-out" if val_ds else " | NO held-out set"))
show(torch.stack([train_ds[i] for i in torch.randperm(len(train_ds))[:16].tolist()]), title="Random training slices")