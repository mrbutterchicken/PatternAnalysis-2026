"""
DICOM -> StyleGAN-ready pipeline for sagittal brain MRI (v2).

Stage 1 (run once): convert_dicoms(src_dir, dst_dir, ...)
    * scans headers only first and groups files into series (one series = one volume)
    * keeps sagittal MR series only; sorts slices by physical position
    * labels each series T1 / T2 / PD (description regex first, TR/TE/TI fallback)
    * standardises orientation (nose on the left, top of head up)
    * normalises intensity PER VOLUME (percentile window on head voxels)
    * trims to a fraction of the HEAD EXTENT along the left-right axis
    * resamples to a common pixel spacing, places on a fixed-FOV canvas, resizes
    * writes 8-bit PNGs into dst/<weighting>/, plus manifest.csv and dataset.json
      (StyleGAN2-ADA label format)

    Run with report_only=True first: it only prints/writes the manifest so you
    can check that the weighting labels make sense before converting anything.

Stage 2 (training): MRISliceDataset / make_loader
    Fast PNG-backed dataset -> tensors [C, H, W] in [-1, 1]. No flip augmentation
    (a left-right flip of a sagittal slice swaps front and back).

pip install pydicom pillow numpy torch tqdm
(+ pylibjpeg / python-gdcm if your files use compressed transfer syntaxes)
Limitation: classic one-slice-per-file DICOM only; enhanced multi-frame files are skipped.
"""
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Checked in order, first match wins (so "T2 MPRAGE" is labelled T2, not T1).
DEFAULT_PATTERNS = {
    "T2": r"T2",
    "PD": r"(?<![A-Za-z])PD(?![A-Za-z])|proton",
    "T1": r"T1|MPRAGE|MP-RAGE",
}


# --------------------------------------------------------------------------
# Header handling
# --------------------------------------------------------------------------
def _num(ds, name):
    try:
        v = ds.get(name, None)
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def _short_hash(value, salt="", n=8):
    return hashlib.sha1((salt + str(value)).encode()).hexdigest()[:n]


def classify_weighting(desc, tr, te, ti, patterns):
    """Return (label, how). Description regex wins; TR/TE/TI is a rough fallback."""
    for label, pat in patterns.items():
        if re.search(pat, desc or "", re.I):
            return label, "description"
    if ti:
        return "T1", "TI present (heuristic)"
    if te is not None:
        if te >= 60:
            return "T2", "TE >= 60 ms (heuristic)"
        if tr is not None and tr >= 2000 and te < 40:
            return "PD", "long TR, short TE (heuristic)"
        if tr is not None and tr < 1500:
            return "T1", "short TR (heuristic)"
    return "unknown", "no match"


def geometry(ds):
    """Row/column direction cosines, slice normal (sign-fixed) and plane name."""
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    r, c = iop[:3], iop[3:]  # r: direction of increasing column idx, c: of increasing row idx
    n = np.cross(r, c)
    plane = ("sagittal", "coronal", "axial")[int(np.argmax(np.abs(n)))]
    if n[0] < 0:
        n = -n
    return r, c, n, plane


def scan_series(src_dir):
    """Group single-frame MR files by SeriesInstanceUID using headers only."""
    series, skipped = defaultdict(list), Counter()
    for p in tqdm(sorted(Path(src_dir).rglob("*.dcm")), desc="Scanning headers"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            skipped["unreadable"] += 1
            continue
        if ds.get("Modality") != "MR":
            skipped["not MR"] += 1
        elif int(ds.get("NumberOfFrames", 1) or 1) > 1:
            skipped["enhanced multi-frame"] += 1
        elif "ImageOrientationPatient" not in ds or "ImagePositionPatient" not in ds:
            skipped["no geometry tags"] += 1
        else:
            series[str(ds.SeriesInstanceUID)].append((p, ds))
    if skipped:
        print("Skipped files:", dict(skipped))
    return series


# --------------------------------------------------------------------------
# Pixel handling
# --------------------------------------------------------------------------
def read_slice(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError("not a single-frame greyscale image")
    return arr * float(ds.get("RescaleSlope", 1.0)) + float(ds.get("RescaleIntercept", 0.0))


def standardize_orientation(vol, r, c, row_sp, col_sp):
    """[N,H,W] -> columns run toward posterior (nose on the left), rows run
    toward inferior (top of image = top of head). Pixel spacings follow the axes."""
    if abs(r[1]) < abs(r[2]):  # columns run along z instead of y -> transpose
        vol = vol.transpose(0, 2, 1)
        r, c = c, r
        row_sp, col_sp = col_sp, row_sp
    if r[1] < 0:  # want columns to increase toward posterior (+y in LPS)
        vol = vol[:, :, ::-1]
    if c[2] > 0:  # want rows to increase toward inferior (-z in LPS)
        vol = vol[:, ::-1, :]
    return np.ascontiguousarray(vol), row_sp, col_sp


def normalize_volume(vol, lo_pct=0.5, hi_pct=99.5):
    """One intensity window for the whole volume, so slice-to-slice brightness
    relationships survive. The upper bound is taken from head voxels only."""
    vol = vol - vol.min()
    ref = np.percentile(vol, 99)
    head = vol[vol > 0.1 * ref]
    if head.size < 1000:
        return None
    lo, hi = np.percentile(vol, lo_pct), np.percentile(head, hi_pct)
    if hi <= lo:
        return None
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def select_slices(vol, keep_range, min_foreground):
    """Indices to keep: a fraction of the head extent along the slice axis
    (slices are sorted by position, so fractions mean the same thing in every
    volume), minus slices that are nearly empty."""
    fg = (vol > 0.1).reshape(len(vol), -1).mean(1)
    ext = np.where(fg >= 0.05 * fg.max())[0]
    if ext.size == 0:
        return []
    e0, e1 = int(ext.min()), int(ext.max()) + 1
    span = e1 - e0
    a = e0 + int(round(keep_range[0] * span))
    b = e0 + int(round(keep_range[1] * span))
    return [i for i in range(a, b) if fg[i] >= min_foreground]


def resample_to_canvas(vol, row_sp, col_sp, target_sp, fov_px, size):
    """Resample to target_sp mm/pixel, centre on a fov_px x fov_px canvas
    (pad or crop), then resize to size x size. Returns float32 [N, size, size]."""
    t = torch.from_numpy(vol).float()[:, None]
    nh = max(1, round(vol.shape[1] * row_sp / target_sp))
    nw = max(1, round(vol.shape[2] * col_sp / target_sp))
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)

    def span(n):
        off = (fov_px - n) // 2
        s0, d0 = max(0, -off), max(0, off)
        return s0, d0, min(n - s0, fov_px - d0)

    sy, dy, ly = span(nh)
    sx, dx, lx = span(nw)
    canvas = torch.zeros(t.shape[0], 1, fov_px, fov_px)
    canvas[:, :, dy:dy + ly, dx:dx + lx] = t[:, :, sy:sy + ly, sx:sx + lx]
    if fov_px != size:
        canvas = F.interpolate(canvas, size=(size, size), mode="bilinear",
                               antialias=True, align_corners=False)
    return canvas[:, 0].clamp(0, 1).numpy()


# --------------------------------------------------------------------------
# Stage 1: DICOM -> PNG
# --------------------------------------------------------------------------
def convert_dicoms(
    src_dir,
    dst_dir,
    size=256,
    target_spacing=1.0,        # mm per pixel after resampling
    fov_mm=256.0,              # fixed field of view -> consistent head scale
    keep_range=(0.35, 0.65),   # fraction of head extent to keep (e.g. slices 60-150 of 200)
    min_foreground=0.05,       # also drop near-empty slices
    min_slices=60,             # ignore localizers / tiny series
    weighting_patterns=None,   # {"T1": regex, ...}; see DEFAULT_PATTERNS
    weightings=None,           # e.g. ["T1"] to convert only some weightings
    report_only=False,         # True: write manifest only, no images
    salt="mri",                # salt for the anonymous subject/series hashes
):
    patterns = weighting_patterns or DEFAULT_PATTERNS
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    fov_px = int(round(fov_mm / target_spacing))
    series = scan_series(src_dir)

    rows, labels = [], []
    for uid, files in tqdm(series.items(), desc="Series"):
        ds0 = files[0][1]
        r, c, n, plane = geometry(ds0)
        tr, te, ti = _num(ds0, "RepetitionTime"), _num(ds0, "EchoTime"), _num(ds0, "InversionTime")
        desc = str(ds0.get("SeriesDescription", ""))
        label, how = classify_weighting(desc, tr, te, ti, patterns)
        subject = _short_hash(ds0.get("PatientID", uid), salt)
        series_id = _short_hash(uid, salt)
        row = dict(subject=subject, series=series_id, description=desc, plane=plane,
                   TR=tr, TE=te, TI=ti, n_files=len(files), weighting=label,
                   labelled_by=how, n_kept=0, status="")

        try:
            if plane != "sagittal":
                row["status"] = "skipped: not sagittal"
            elif len(files) < min_slices:
                row["status"] = "skipped: too few slices"
            elif weightings and label not in weightings:
                row["status"] = "skipped: weighting not requested"
            elif report_only:
                row["status"] = "report only"
            else:
                files.sort(key=lambda f: float(np.dot(np.array(f[1].ImagePositionPatient, float), n)))
                shape = Counter((int(d.Rows), int(d.Columns)) for _, d in files).most_common(1)[0][0]
                files = [f for f in files if (int(f[1].Rows), int(f[1].Columns)) == shape]

                vol = np.stack([read_slice(p) for p, _ in files])
                if ds0.get("PhotometricInterpretation") == "MONOCHROME1":
                    vol = vol.max() - vol
                ps = ds0.get("PixelSpacing", [1.0, 1.0])
                vol, row_sp, col_sp = standardize_orientation(vol, r, c, float(ps[0]), float(ps[1]))

                vol = normalize_volume(vol)
                idx = select_slices(vol, keep_range, min_foreground) if vol is not None else []
                if not idx:
                    row["status"] = "skipped: no usable slices"
                else:
                    imgs = resample_to_canvas(vol[idx], row_sp, col_sp, target_spacing, fov_px, size)
                    out = dst / label
                    out.mkdir(exist_ok=True)
                    for i, img in zip(idx, imgs):
                        name = f"{subject}_{series_id}_{i:03d}.png"
                        Image.fromarray((img * 255).round().astype(np.uint8)).save(out / name)
                        labels.append((f"{label}/{name}", label))
                    row["n_kept"], row["status"] = len(idx), "converted"
        except Exception as e:
            row["status"] = f"failed: {e}"
        rows.append(row)

    if rows:
        with open(dst / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if labels:
        classes = sorted({lab for _, lab in labels})
        with open(dst / "dataset.json", "w") as f:  # StyleGAN2-ADA label format
            json.dump({"labels": [[fn, classes.index(lab)] for fn, lab in labels]}, f)

    print(f"{len(rows)} series: ", dict(Counter(r["status"].split(":")[0] for r in rows)))
    print("Weightings:", dict(Counter(r["weighting"] for r in rows)))
    print(f"{len(labels)} slices written -> {dst}   (see manifest.csv)")


# --------------------------------------------------------------------------
# Stage 2: training-time dataset
# --------------------------------------------------------------------------
def split_subjects(root, val_frac=0.1, seed=0):
    """Subject-level split (filenames start with an anonymous subject hash), so
    slices of one person never land in both train and validation."""
    subjects = sorted({f.name.split("_")[0] for f in Path(root).rglob("*.png")})
    random.Random(seed).shuffle(subjects)
    k = max(1, int(round(len(subjects) * val_frac)))
    return set(subjects[k:]), set(subjects[:k])


class MRISliceDataset(Dataset):
    """PNG slices -> float tensors [C, H, W] in [-1, 1].

    weightings:   e.g. ["T1"] to train on one contrast only (folder names)
    subjects:     optional set of subject hashes (from split_subjects)
    return_label: also return a one-hot weighting vector (conditional GAN)
    channels:     1, or 3 if your StyleGAN code insists on RGB
    """

    def __init__(self, root, weightings=None, subjects=None, return_label=False, channels=1):
        files = sorted(Path(root).rglob("*.png"))
        self.classes = sorted({f.parent.name for f in files})
        if weightings:
            files = [f for f in files if f.parent.name in weightings]
        if subjects is not None:
            files = [f for f in files if f.name.split("_")[0] in subjects]
        if not files:
            raise FileNotFoundError(f"No matching PNGs under {root}")
        self.files, self.return_label, self.channels = files, return_label, channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.files[i]).convert("L"), dtype=np.float32)
        x = torch.from_numpy(img / 127.5 - 1.0).unsqueeze(0)
        if self.channels == 3:
            x = x.repeat(3, 1, 1)
        if self.return_label:
            y = torch.zeros(len(self.classes))
            y[self.classes.index(self.files[i].parent.name)] = 1.0
            return x, y
        return x


def make_loader(root, batch_size=32, num_workers=4, **ds_kwargs):
    return DataLoader(
        MRISliceDataset(root, **ds_kwargs),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=True, persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--fov", type=float, default=256.0)
    ap.add_argument("--keep", type=float, nargs=2, default=(0.30, 0.75))
    ap.add_argument("--weightings", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    convert_dicoms(a.src, a.dst, size=a.size, target_spacing=a.spacing, fov_mm=a.fov,
                   keep_range=tuple(a.keep), weightings=a.weightings, report_only=a.report_only)