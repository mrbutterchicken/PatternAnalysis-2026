# =============================================================================
# SETUP & CONFIG   (run first; edit the values below as needed)
# =============================================================================
import copy, csv, json, math, os, random, re, time, zipfile
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
# DATA_SOURCE can be: a .zip, a folder containing the .zip (fastest: one file to read from Drive), or the dataset
# folder itself (the one that contains train/ and test/; slower, because Drive serves thousands of small files).
DATA_SOURCE = "/content/drive/MyDrive/AD_NC.zip"
UNZIP_TO    = "/content"
OUT_DIR     = "/content/drive/MyDrive/stylegan-sagittal"   # checkpoints, samples, logs (on Drive: survives disconnects)

# ---- data -----------------------------------------------------------------------------------
RES_TARGET    = 256     # images are padded to a square and resized to this (power of two)
CLASSES       = None    # None = pool AD and NC into one unconditional dataset; or e.g. ["NC"] to train on one class
SUBJECT_REGEX = r"^(\d+)_\d+"   # first group = patient number:   '218391_78.jpeg' -> 218391
SLICE_REGEX   = r"^\d+_(\d+)"   # first group = slice number:     '218391_78.jpeg' -> 78
SLICE_RANGE   = None    # e.g. (40, 140) keeps only those slice numbers; look at the slice plots from DATASET first
MAX_HELD_OUT  = 2000    # at most this many held-out images are loaded for the PREDICT audits
MIN_HELD_OUT  = 200     # if fewer than this many test/ images belong to patients unseen in train/, test/ is treated as leaky ...
HELD_OUT_FRAC = 0.05    # ... and this fraction of the training PATIENTS is held out of training instead
SEED          = 0

# ---- model ----------------------------------------------------------------------------------
Z_DIM = W_DIM = 512
MAP_LAYERS = 4                         # StyleGAN2 uses 8; 4 is plenty for one narrow domain
CHANNEL_BASE, MAX_CH = 16384, 512      # channels at resolution r = min(CHANNEL_BASE // r, MAX_CH)

# ---- training -------------------------------------------------------------------------------
BATCH          = 32
EPOCHS         = 60        # 1 epoch = one pass over the training images (~21k images = ~660 iterations)
LR             = 0.002
R1_GAMMA       = 10.0      # R1 gradient-penalty strength (applied lazily, every R1_EVERY steps)
R1_EVERY       = 16
STYLE_MIX_PROB = 0.9
EMA_KIMG       = 10        # half-life (thousands of images) of the EMA generator used for sampling
ADA_TARGET, ADA_KIMG = 0.6, 100    # adaptive augmentation: target mean sign(D(real)), adjustment speed
SAVE_EVERY     = 5         # epochs between checkpoints
LOG_EVERY      = 50        # iterations between log lines
RESUME         = True      # continue from latest.pt if it exists
QUICK_TEST     = False     # True = two tiny epochs to check the whole pipeline end to end (re-run this cell after changing)

RUN_DIR = Path(OUT_DIR) / ("quicktest" if QUICK_TEST else "run")
(RUN_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "samples").mkdir(exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =============================================================================
# DATASET   brain-only JPEG slices: train/{AD,NC}, test/{AD,NC}  ->  RAM as uint8  ->  fast GPU batches
#           (AD and NC are pooled: one unconditional model)
# =============================================================================
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageOps

IMG_EXTS = (".jpg", ".jpeg")


def locate_root():
    """Folder that contains train/ (and test/). Unzips first if the data is zipped."""
    src = Path(DATA_SOURCE)
    if src.is_dir() and (src / "train").is_dir():
        return src
    zp = src if src.is_file() else None
    if zp is None:
        zips = sorted(src.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if not zips:
            raise FileNotFoundError(f"No dataset found at {src}: expected a .zip, or a folder containing train/ "
                                    f"(on Colab the Drive folder is 'MyDrive', capital M and D)")
        zp = zips[-1]
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith(IMG_EXTS) and "__MACOSX" not in n and not Path(n).name.startswith("._")]
        todo = [n for n in names if not (Path(UNZIP_TO) / n).exists()]
        if todo:
            print(f"Extracting {len(todo)} images to {UNZIP_TO} ...")
            z.extractall(UNZIP_TO, todo)
    for n in names:                                   # the folder that holds train/ is the dataset root
        parts = Path(n).parts
        if "train" in parts:
            return Path(UNZIP_TO).joinpath(*parts[:parts.index("train")])
    raise RuntimeError("No train/ folder found inside the zip")


def list_split(root, split):
    folder = root / split
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else []
    if CLASSES:
        files = [f for f in files if f.parent.name in CLASSES]
    if SLICE_RANGE:
        files = [f for f in files if SLICE_RANGE[0] <= slice_of(f) <= SLICE_RANGE[1]]
    counts = defaultdict(int)
    for f in files:
        counts[f.parent.name] += 1
    print(f"{split}/: {len(files)} images {dict(counts)}")
    return files


def subject_of(path):
    """Patient id from the filename (SUBJECT_REGEX). Used for the held-out split and the PREDICT audits."""
    m = re.search(SUBJECT_REGEX, Path(path).name)
    if not m:
        raise ValueError(f"{Path(path).name} does not match SUBJECT_REGEX")
    return m.group(1)


def slice_of(path):
    m = re.search(SLICE_REGEX, Path(path).name)
    if not m:
        raise ValueError(f"{Path(path).name} does not match SLICE_REGEX")
    return int(m.group(1))


def load_square(path, res=RES_TARGET):
    """JPEG -> greyscale -> padded to a black square (aspect ratio preserved) -> res x res uint8 array."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(canvas.resize((res, res), Image.LANCZOS) if s != res else canvas)


class MRISlices(torch.utils.data.Dataset):
    """Every slice held in RAM as uint8 [N,1,H,W]; items come out as float [1,H,W] in [-1,1]."""
    def __init__(self, files):
        self.files = list(files)
        with ThreadPoolExecutor(8) as ex:
            imgs = list(tqdm(ex.map(load_square, self.files), total=len(self.files), desc="Loading images"))
        self.data = torch.from_numpy(np.stack(imgs))[:, None].contiguous()
        self.res = RES_TARGET
        self.subjects = [subject_of(f) for f in self.files]
        self.labels = [f.parent.name for f in self.files]       # AD / NC: not used for training, handy for later evaluation
        self.slices = [slice_of(f) for f in self.files]

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
    """train/ trains the model. The held-out set must contain PATIENTS the model never trained on, so we check
    which test/ patients also appear in train/; if test/ is mostly leaky we hold out some training patients instead."""
    root = locate_root()
    train_files, test_files = list_split(root, "train"), list_split(root, "test")
    if not train_files:
        raise RuntimeError(f"No JPEGs found in {root / 'train'}")
    train_subj = {subject_of(f) for f in train_files}
    test_subj = {subject_of(f) for f in test_files}
    print(f"{len(train_subj)} patients in train/, {len(test_subj)} in test/, {len(train_subj & test_subj)} appear in BOTH")

    val_files = [f for f in test_files if subject_of(f) not in train_subj]
    if len(val_files) >= MIN_HELD_OUT:
        print(f"Held-out set = the {len(val_files)} test/ images from patients unseen in train/")
    else:
        ids = sorted(train_subj)
        random.Random(SEED).shuffle(ids)
        held = set(ids[:max(1, round(len(ids) * HELD_OUT_FRAC))])
        val_files = [f for f in train_files if subject_of(f) in held]
        train_files = [f for f in train_files if subject_of(f) not in held]
        print(f"test/ shares patients with train/ -> held out {len(held)} TRAINING patients ({len(val_files)} images) instead")
    if len(val_files) > MAX_HELD_OUT:
        val_files = random.Random(SEED).sample(val_files, MAX_HELD_OUT)

    per_patient = np.bincount(np.unique([subject_of(f) for f in train_files], return_inverse=True)[1])
    print(f"training images per patient: min {per_patient.min()}, median {int(np.median(per_patient))}, max {per_patient.max()}")
    return MRISlices(train_files), MRISlices(sorted(val_files)) if val_files else None


def inspect_slices(ds):
    """How many images per slice number, and how large the brain is at each slice number."""
    sl = np.array(ds.slices)
    area = np.concatenate([(ds.data[i:i + 512] > 20).float().mean((1, 2, 3)).numpy() for i in range(0, len(ds), 512)])
    idx = sl - sl.min()
    cnt = np.bincount(idx)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    ax[0].bar(np.arange(sl.min(), sl.max() + 1), cnt); ax[0].set_title("training images per slice number")
    ax[1].plot(np.arange(sl.min(), sl.max() + 1), np.bincount(idx, weights=area) / np.maximum(cnt, 1))
    ax[1].set_title("mean fraction of non-black pixels (brain size) per slice number")
    for a in ax:
        a.set_xlabel("slice number")
    plt.tight_layout(); plt.show()


train_ds, val_ds = prepare_data()
RES = train_ds.res
assert len(train_ds) >= BATCH, f"only {len(train_ds)} training images, fewer than BATCH={BATCH}"
print(f"{len(train_ds)} training images at {RES}x{RES}" + (f" | {len(val_ds)} held-out (test/)" if val_ds else " | NO held-out set"))
show(torch.stack([train_ds[i] for i in torch.randperm(len(train_ds))[:16].tolist()]), title="Random training images (AD and NC pooled)")
inspect_slices(train_ds)