"""
Showing example usage of your **trained** model. It should load your saved model
weights, run inference on test cases, print out any results, and provide 
visualisations where applicable (e.g., prediction overlays, generated samples, 
heatmaps). NumPy is allowed in this script for visualization or loading data.
"""

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon (M1/M2/M3/M4)
else:
    device = torch.device("cpu")        # Fallback Default

print(f"Using device: {device}")

import numpy as np
from matplotlib import pyplot as plt

from dataset import *
from modules import *

# =============================================================================
# PREDICT   needs the SETUP, DATASET and MODULES cells (TRAINING is not required in a fresh session)
# =============================================================================
import torchvision
from scipy import stats as sps
from scipy.linalg import sqrtm

CKPT = RUN_DIR / "latest.pt"          # or e.g. RUN_DIR / "snapshots" / "G_ema_epoch0100.pt"
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
G_ema = Generator(ck["res"]).to(DEVICE)
G_ema.load_state_dict(ck["G_ema"])
G_ema.eval().requires_grad_(False)
print(f"Loaded {CKPT.name} (epoch {ck['epoch']})")

AUDIT_DIR = RUN_DIR / "audit"
AUDIT_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def generate(n, psi=1.0, seed=0, noise_mode="random", bs=100):
    """n synthetic slices in [-1,1] on the CPU. psi < 1 pulls styles toward the average style (cleaner, less diverse)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        z = torch.randn(min(bs, n - i), Z_DIM, generator=g).to(DEVICE)
        out.append(G_ema(z, psi=psi, noise_mode=noise_mode).clamp(-1, 1).cpu())
    return torch.cat(out)


show(generate(32, 1.0, seed=1), nrow=8, title="Synthetic slices, psi = 1.0 (full diversity)")
show(generate(32, 0.7, seed=1), nrow=8, title="Synthetic slices, psi = 0.7 (truncated: cleaner but less diverse)")

# ---- 1. NEAREST-NEIGHBOUR AUDIT: are synthetic slices copies of training slices? -------------------
# Three distance distributions, all measured against the TRAINING set:
#   synthetic -> train                      (what we are testing)
#   held-out real -> train                  (how far a genuinely NEW scan sits from the training data)
#   train -> train, other subjects only     (natural similarity between different people's slices)
# "Near-copy" = closer to a training slice than 99% of training slices are to any OTHER person's slice.
N_GEN = 200 if QUICK_TEST else 1000
gen = generate(N_GEN, psi=1.0, seed=42)
sets = {"train": train_ds.data, "gen": gen}
if val_ds is not None:
    sets["val"] = val_ds.data


def prep(b):
    """uint8 [0,255] or float [-1,1]  ->  float [0,1] on the GPU"""
    b = b.to(DEVICE)
    return b.float() / 255 if b.dtype == torch.uint8 else (b.float().clamp(-1, 1) + 1) / 2


@torch.no_grad()
def pixfeat(x, bs=512, size=64):
    return torch.cat([F.interpolate(prep(x[i:i + bs]), size=(size, size), mode="area").flatten(1)
                      for i in range(0, len(x), bs)])


_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
_resnet.fc = nn.Identity()
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def embed(x, bs=256):
    out = []
    for i in range(0, len(x), bs):
        b = F.interpolate(prep(x[i:i + bs]), size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        out.append(_resnet((b - _MEAN) / _STD).cpu())
    return torch.cat(out)


@torch.no_grad()
def nn_search(Q, R, q_sid=None, r_sid=None, chunk=512):
    """Nearest neighbour in R for every row of Q; with subject ids, same-subject matches are ignored."""
    dist, idx = [], []
    for i in range(0, len(Q), chunk):
        d = torch.cdist(Q[i:i + chunk], R)
        if q_sid is not None:
            d[q_sid[i:i + chunk, None] == r_sid[None, :]] = float("inf")
        v, j = d.min(1)
        dist.append(v); idx.append(j)
    return torch.cat(dist).cpu().numpy(), torch.cat(idx).cpu().numpy()


print("Computing pixel and ResNet-18 features ...")
P = {k: pixfeat(v) for k, v in sets.items()}                       # 64x64 pixel vectors
E = {k: embed(v) for k, v in sets.items()}                         # raw ResNet-18 embeddings (CPU)
En = {k: F.normalize(v.to(DEVICE), dim=1) for k, v in E.items()}   # unit-length copies for distances

vocab = {s: i for i, s in enumerate(sorted(set(train_ds.subjects)))}
train_sid = torch.tensor([vocab[s] for s in train_ds.subjects], device=DEVICE)
can_cross = len(vocab) >= 2                                        # need >= 2 training subjects for the reference
sel = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(0))[:2000].to(DEVICE)

NN = {}
for name, (feats, scale) in {"pixel": (P, 1 / math.sqrt(P["train"].shape[1])), "resnet": (En, 1.0)}.items():
    d_gen, i_gen = nn_search(feats["gen"], feats["train"])
    d_val = nn_search(feats["val"], feats["train"])[0] * scale if val_ds is not None else None
    d_ref = nn_search(feats["train"][sel], feats["train"], train_sid[sel], train_sid)[0] * scale if can_cross else None
    d_gen = d_gen * scale
    thr = float(np.percentile(d_ref, 1)) if can_cross else None
    NN[name] = dict(d_gen=d_gen, i_gen=i_gen, d_val=d_val, d_ref=d_ref, thr=thr,
                    copy_rate=float((d_gen < thr).mean()) if can_cross else None,
                    copy_rate_val=float((d_val < thr).mean()) if (can_cross and d_val is not None) else None)
    r = NN[name]
    print(f"\n[{name}] median nearest-neighbour distance to the training set")
    print(f"   synthetic          : {np.median(d_gen):.4f}")
    if d_val is not None:
        print(f"   held-out real      : {np.median(d_val):.4f}")
    if can_cross:
        print(f"   train (other subj.): {np.median(d_ref):.4f}")
        print(f"   near-copy threshold (1st percentile of the line above): {thr:.4f}")
        print(f"   synthetic slices below it: {100 * r['copy_rate']:.1f}%"
              + (f"   | held-out real slices below it: {100 * r['copy_rate_val']:.1f}%" if r["copy_rate_val"] is not None else ""))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for a, (name, r) in zip(ax, NN.items()):
    a.hist(r["d_gen"], bins=50, alpha=0.6, density=True, label="synthetic -> train")
    if r["d_val"] is not None:
        a.hist(r["d_val"], bins=50, alpha=0.5, density=True, label="held-out real -> train")
    if r["d_ref"] is not None:
        a.hist(r["d_ref"], bins=50, alpha=0.4, density=True, label="train -> other-subject train")
        a.axvline(r["thr"], color="k", ls="--", label="near-copy threshold")
    a.set_title(f"NN distance, {name} space"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(AUDIT_DIR / "nn_distances.png", dpi=110); plt.show()

# the 8 synthetic slices that sit closest to the training set (worst case), next to their nearest training slice
order = np.argsort(NN["pixel"]["d_gen"])[:8]
fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
for c_, gi in enumerate(order):
    ax[0, c_].imshow(((gen[gi, 0] + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
    ax[0, c_].set_title(f"RMS {NN['pixel']['d_gen'][gi]:.3f}", fontsize=8)
    ax[1, c_].imshow(train_ds.data[NN["pixel"]["i_gen"][gi], 0].numpy(), cmap="gray", vmin=0, vmax=255)
    ax[0, c_].axis("off"); ax[1, c_].axis("off")
fig.text(0.005, 0.72, "synthetic", rotation=90, va="center"); fig.text(0.005, 0.28, "nearest train", rotation=90, va="center")
plt.tight_layout(); plt.savefig(AUDIT_DIR / "closest_pairs.png", dpi=110); plt.show()

# ---- 2. PROJECTION + LATENT INTERPOLATION: synthetic image <-> real scan ---------------------------
# To interpolate between a real scan and a synthetic image, the real scan first has to be placed in latent
# space ("projection": optimise a W+ code until G reproduces it). This doubles as a memorisation test:
# if G memorised its training scans it would reproduce TRAINING scans far better than unseen HELD-OUT ones.
PROJ_STEPS = 30 if QUICK_TEST else 400


def pyramid_mse(a, b):
    loss = 0
    for s in (1, 2, 4, 8):
        a_, b_ = (F.avg_pool2d(a, s), F.avg_pool2d(b, s)) if s > 1 else (a, b)
        loss = loss + (a_ - b_).square().mean([1, 2, 3])
    return loss                                                    # [B]


def project(targets, steps=PROJ_STEPS, lr=0.1):
    """One W+ code per target image (zero noise, so reconstructions are deterministic)."""
    targets = targets.to(DEVICE)
    w = G_ema.w_avg[None, None].repeat(len(targets), G_ema.num_ws, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    for i in tqdm(range(steps), desc="projecting", leave=False):
        for g_ in opt.param_groups:
            g_["lr"] = lr * 0.5 * (1 + math.cos(math.pi * i / steps))
        loss = pyramid_mse(G_ema.synthesis(w, noise_mode="zero"), targets).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        recon = G_ema.synthesis(w, noise_mode="zero").clamp(-1, 1)
    rmse = ((recon - targets) / 2).square().mean([1, 2, 3]).sqrt().cpu().numpy()      # error on a 0-1 intensity scale
    return w.detach(), recon.cpu(), rmse


g_ = torch.Generator().manual_seed(7)
K = 8
tr_idx = torch.randperm(len(train_ds), generator=g_)[:K].tolist()
tr_imgs = torch.stack([train_ds[i] for i in tr_idx])
if val_ds is not None:
    va_idx = torch.randperm(len(val_ds), generator=g_)[:K].tolist()
    va_imgs = torch.stack([val_ds[i] for i in va_idx])
    imgs, n_tr = torch.cat([tr_imgs, va_imgs]), len(tr_imgs)
else:
    imgs, n_tr = tr_imgs, len(tr_imgs)

w_all, recon_all, rmse_all = project(imgs)
rmse_train = rmse_all[:n_tr]
rmse_val = rmse_all[n_tr:] if val_ds is not None else None
print(f"Projection error (RMSE, 0-1 scale): training scans {rmse_train.mean():.4f}"
      + (f" | held-out scans {rmse_val.mean():.4f} | ratio held-out/train {rmse_val.mean() / rmse_train.mean():.2f}" if rmse_val is not None else ""))

# rows: 4 training scans (+ 2 held-out scans), each interpolated toward a different random synthetic image
rows = list(range(min(4, n_tr))) + (list(range(n_tr, n_tr + min(2, len(imgs) - n_tr))) if val_ds is not None else [])
labels = ["training scan" if r < n_tr else "held-out scan" for r in rows]
T = 8
ts = torch.linspace(0, 1, T, device=DEVICE)
with torch.no_grad():
    ws_syn = G_ema.make_ws(torch.randn(len(rows), Z_DIM, generator=torch.Generator().manual_seed(11)).to(DEVICE))
    frames = []
    for j, r in enumerate(rows):
        path = (1 - ts)[:, None, None] * w_all[r:r + 1] + ts[:, None, None] * ws_syn[j:j + 1]    # linear walk in W+
        frames.append(G_ema.synthesis(path, noise_mode="zero").clamp(-1, 1).cpu())

fig, ax = plt.subplots(len(rows), T + 1, figsize=(1.7 * (T + 1), 1.7 * len(rows)), squeeze=False)
for j, r in enumerate(rows):
    ax[j, 0].imshow(((imgs[r, 0] + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
    ax[j, 0].text(-0.06, 0.5, labels[j], transform=ax[j, 0].transAxes, ha="right", va="center", fontsize=8)
    for t in range(T):
        ax[j, t + 1].imshow(((frames[j][t, 0] + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
    for a in ax[j]:
        a.axis("off")
ax[0, 0].set_title("original", fontsize=9); ax[0, 1].set_title("t=0 projected", fontsize=9)
ax[0, T].set_title("t=1 synthetic", fontsize=9)
plt.tight_layout(); plt.savefig(AUDIT_DIR / "interpolation.png", dpi=110, bbox_inches="tight"); plt.show()

# smoothness of each path: size of the largest frame-to-frame change relative to the average change
jump = []
for f in frames:
    step = ((f[1:] - f[:-1]) / 2).square().mean([1, 2, 3]).sqrt()
    jump.append((step.max() / step.mean().clamp(min=1e-8)).item())
jump_ratio = float(max(jump))
print(f"Interpolation smoothness (largest step / mean step, worst row): {jump_ratio:.2f}  (1 = perfectly even)")

# ---- 3. PLAUSIBILITY PROXIES + VERDICT --------------------------------------------------------------
# These are automatic stand-ins, NOT a clinical review. Thresholds are rules of thumb, not validated standards.
def frechet(a, b):
    a, b = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(sqrtm(s1 @ s2))
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


@torch.no_grad()
def img_stats(x, bs=500):
    out = defaultdict(list)
    for i in range(0, len(x), bs):
        b = prep(x[i:i + bs])[:, 0]
        H = b.shape[1]
        fg = (b > 0.15).float()
        n = fg.sum((1, 2)).clamp(min=1)
        idx = torch.arange(H, device=DEVICE).float() / H
        for k, v in dict(area=fg.mean((1, 2)), mean_fg=(b * fg).sum((1, 2)) / n,
                         cy=(fg * idx.view(1, H, 1)).sum((1, 2)) / n, cx=(fg * idx.view(1, 1, H)).sum((1, 2)) / n).items():
            out[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def radial_psd(x, bs=250):
    acc = 0
    for i in range(0, len(x), bs):
        b = prep(x[i:i + bs])[:, 0]
        b = b - b.mean((1, 2), keepdim=True)
        acc = acc + torch.fft.fftshift(torch.fft.fft2(b), dim=(-2, -1)).abs().square().sum(0)
    acc = acc / len(x)
    H = acc.shape[0]
    ax_ = torch.arange(H, device=acc.device) - H // 2
    r = (ax_.view(-1, 1) ** 2 + ax_.view(1, -1) ** 2).float().sqrt().long().flatten()
    prof = torch.bincount(r, weights=acc.flatten()) / torch.bincount(r)
    return prof[:H // 2].cpu().numpy()


ref_idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(1))[:min(2000, len(train_ds))]
ref_train = train_ds.data[ref_idx]

# (a) Frechet distance in ResNet-18 feature space, calibrated against a real-vs-real baseline
fd_ratio = None
if val_ds is not None and len(val_ds) >= 50:
    n = min(len(E["gen"]), len(E["val"]), 2000)
    fd_gen = frechet(E["gen"][:n], E["val"][:n])
    fd_base = frechet(E["train"][ref_idx[:n]], E["val"][:n])
    fd_ratio = fd_gen / max(fd_base, 1e-8)
    print(f"Frechet distance to held-out real (n={n}): synthetic {fd_gen:.2f} | training slices {fd_base:.2f} | ratio {fd_ratio:.2f}"
          + ("   (small n: only the ratio is meaningful)" if n < 500 else ""))

# (b) anatomy-level statistics: head area, mean tissue brightness, head position (vs the training slices)
st_g, st_r = img_stats(gen), img_stats(ref_train)
ks = {k: float(sps.ks_2samp(st_g[k], st_r[k]).statistic) for k in st_g}
print("KS distance synthetic vs training (0 = identical):", {k: round(v, 3) for k, v in ks.items()})

# (c) frequency content: GANs sometimes leave grid artefacts or over-smooth the noise texture
psd_g, psd_r = radial_psd(gen), radial_psd(ref_train)
band = slice(len(psd_g) // 2, len(psd_g))
psd_diff = float(np.mean(np.abs(np.log10(psd_g[band]) - np.log10(psd_r[band]))))
psd_sign = float(np.mean(np.log10(psd_g[band]) - np.log10(psd_r[band])))
plt.figure(figsize=(5, 3.4))
plt.semilogy(psd_r, label="training"); plt.semilogy(psd_g, label="synthetic")
plt.axvline(band.start, color="k", ls=":"); plt.xlabel("spatial frequency (cycles per image)"); plt.ylabel("power")
plt.title("radially averaged power spectrum"); plt.legend(); plt.tight_layout()
plt.savefig(AUDIT_DIR / "power_spectrum.png", dpi=110); plt.show()
print(f"High-frequency band: |log10 power difference| {psd_diff:.2f} ({'more' if psd_sign > 0 else 'less'} energy than real)")

# ---- verdict ------------------------------------------------------------------------------------
checks = []                                                         # (group, name, value, status, rule)


def grade(group, name, value, ok, warn, fmt="{:.2f}", rule=None):
    if value is None:
        checks.append((group, name, "n/a", "SKIP", "needs held-out subjects" if group == "memorisation" else "not computed"))
        return
    status = "PASS" if value <= ok else "WARN" if value <= warn else "FAIL"
    checks.append((group, name, fmt.format(value), status, rule or f"PASS <= {ok}, WARN <= {warn}"))


for name, r in NN.items():
    grade("memorisation", f"near-copy rate, {name} space", None if r["copy_rate"] is None else 100 * r["copy_rate"],
          5, 20, "{:.1f}%", "PASS <= 5%, WARN <= 20% of synthetic slices")
grade("memorisation", "projection error, held-out / training",
      None if rmse_val is None else float(rmse_val.mean() / rmse_train.mean()), 1.3, 2.0)
grade("plausibility", "Frechet ratio (synthetic vs real-vs-real baseline)", fd_ratio, 1.5, 3.0)
grade("plausibility", "worst KS distance (area / brightness / position)", max(ks.values()), 0.15, 0.30, "{:.3f}")
grade("plausibility", "high-frequency spectrum mismatch (log10)", psd_diff, 0.25, 0.50)
grade("plausibility", "latent-interpolation smoothness", jump_ratio, 3.0, 5.0)

print("\n" + "=" * 100)
print(f"{'check':58s}{'value':>10s}  status  rule")
for grp, nm, val, st, rule in checks:
    print(f"{nm:58s}{val:>10s}  {st:6s}  {rule}")


def worst(group):
    sts = [c[3] for c in checks if c[0] == group and c[3] != "SKIP"]
    return "FAIL" if "FAIL" in sts else "WARN" if "WARN" in sts else "PASS" if sts else "NO DATA"


MEM = {"PASS": "no sign that synthetic slices are copies of training slices: they sit no closer to the training set than genuinely new scans do, and G does not reproduce training scans better than unseen ones.",
       "WARN": "borderline. Some synthetic slices are unusually close to training slices, or training scans are reproduced better than unseen ones. Look at closest_pairs.png, try an earlier snapshot, and add more training subjects if you can.",
       "FAIL": "likely memorisation. Many synthetic slices are near-duplicates of training slices. Use an earlier snapshot, more subjects, stronger augmentation, or less training.",
       "NO DATA": "cannot be assessed (no held-out subjects / too few training subjects)."}
PLA = {"PASS": "the synthetic slices match real data on every automatic proxy (feature distribution, head size / brightness / position, noise texture, smooth latent paths).",
       "WARN": "partly plausible: at least one proxy deviates from real data. Check the sample grids and power spectrum before trusting the images.",
       "FAIL": "not yet plausible: at least one proxy is far from real data (training probably unfinished, or artefacts / collapse).",
       "NO DATA": "no proxy could be computed."}
m, p_ = worst("memorisation"), worst("plausibility")
print("\nVERDICT")
print(f"  Memorisation : {m}: {MEM[m]}")
print(f"  Plausibility : {p_}: {PLA[p_]}")
print("  Caveat       : these are statistical proxies. Whether the images are CLINICALLY plausible (anatomy, tissue contrast,")
print("                 absence of subtle artefacts) can only be judged by a radiologist / neuroimaging expert. Not for diagnostic use.")

with open(AUDIT_DIR / "audit.json", "w") as f:
    json.dump(dict(epoch=ck["epoch"], checks=checks, memorisation=m, plausibility=p_,
                   nn={k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in NN.items()}), f, indent=1, default=float)
print(f"\nFigures and audit.json saved in {AUDIT_DIR}")