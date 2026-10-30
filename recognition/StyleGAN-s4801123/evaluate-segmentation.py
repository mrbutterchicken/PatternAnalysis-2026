
from dataset import *
from modules import *
from evaluate import *

# =============================================================================
# EVALUATION (segmentation)   
#
# A segmenter is a MEASURING INSTRUMENT: it does not have to be perfect, it has to treat real and synthetic images
# the same way. Dice needs two masks of the SAME anatomy, which an unconditional GAN's images do not have, so we use
#   1. class-area statistics (how much CSF / grey / white matter)      
#   2. segmenter confidence and stability
#   3. best-match Dice: is there a REAL mask resembling each synthetic mask as well as one resembles a new real scan?
#   4. paired Dice: real scan vs the generator's reconstruction of it (via projection)
# =============================================================================
CLASS_NAMES = ["background", "CSF", "GM", "WM"] 
TISSUE = [1, 2, 3]                                    # labels that enter the Dice scores (background excluded)
FG_THR = 0.08                                         # pixels brighter than this (0-1 scale) count as brain

# CLASS_NAMES = [f"class_{i}" for i in range(103)]
# TISSUE = list(range(1, 103))  # Exclude background (channel 0)


# ---- numpy mask metrics (masks are small integer arrays [N,H,W]) -------------------------------------
def dice_per_class(a, b, classes):
    """Paired Dice for every image and class. NaN where the class is absent from both masks."""
    out = np.full((len(a), len(classes)), np.nan)
    for j, c in enumerate(classes):
        A, B = (a == c).reshape(len(a), -1), (b == c).reshape(len(b), -1)
        s = A.sum(1) + B.sum(1)
        out[:, j] = np.where(s > 0, 2 * (A & B).sum(1) / np.maximum(s, 1), np.nan)
    return out


def macro_dice(d):
    """Mean over the classes that are present (NaN-aware); NaN if none are."""
    n = np.sum(~np.isnan(d), 1)
    return np.where(n > 0, np.nansum(d, 1) / np.maximum(n, 1), np.nan)


def best_match_dice(Q, R, classes, chunk=250):
    """For every mask in Q the highest macro-Dice against any mask in R (and which one). Q [n,h,w], R [m,h,w]."""
    oR = [(R == c).reshape(len(R), -1).astype(np.float32) for c in classes]
    best, idx = np.zeros(len(Q), np.float32), np.zeros(len(Q), int)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        num, cnt = np.zeros((len(q), len(R)), np.float32), np.zeros((len(q), len(R)), np.float32)
        for c, o in zip(classes, oR):
            oq = (q == c).reshape(len(q), -1).astype(np.float32)
            s = oq.sum(1)[:, None] + o.sum(1)[None, :]
            num += np.where(s > 0, 2 * (oq @ o.T) / np.maximum(s, 1), 0)
            cnt += s > 0
        macro = num / np.maximum(cnt, 1)
        j = macro.argmax(1)
        best[i:i + chunk], idx[i:i + chunk] = macro[np.arange(len(q)), j], j
    return best, idx


def tissue_share(mask):
    """Share of the brain area taken by each tissue class, [N, len(TISSUE)]."""
    brain = np.maximum((mask > 0).reshape(len(mask), -1).sum(1), 1)
    return np.stack([(mask == c).reshape(len(mask), -1).sum(1) / brain for c in TISSUE], 1)


# ---- segmenters ----------------------------------------------------------------------------------------
def as_pm1(b):
    """uint8 [0,255] or float [-1,1]  ->  float [-1,1] on the GPU"""
    b = b.to(DEVICE)
    return b.float() / 127.5 - 1 if b.dtype == torch.uint8 else b.float().clamp(-1, 1)


class TorchSegmenter:
    """Wrap any 2D PyTorch segmentation net (your OASIS model, a pretrained model, one you train yourself).
    model: [B,1,h,w] -> logits [B,C,h,w] with C == len(CLASS_NAMES). `pre` / `post` are optional hooks for tensor
    re-orientation or label remapping. Called with images in [-1,1]; returns class probabilities [B,C,H,W]."""
    def __init__(self, model, size=None, norm="minmax", pre=None, post=None):
        self.model, self.size, self.norm, self.pre, self.post = model.to(DEVICE).eval(), size, norm, pre, post

    @torch.no_grad()
    def __call__(self, x):
        H, W = x.shape[-2:]

        # Convert from [-1, 1] to [0, 1].
        z = (x + 1) / 2

        # DARTSeg's network expects 256 x 256 slices.
        if self.size and z.shape[-2:] != (self.size, self.size):
            z = F.interpolate(
                z,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )

        if self.norm == "minmax":
            # Normalise each image independently, as DARTSeg does.
            lo = z.amin(dim=(-2, -1), keepdim=True)
            hi = z.amax(dim=(-2, -1), keepdim=True)
            z = (z - lo) / (hi - lo + 1e-4)

        elif self.norm == "zscore":
            fg = (z > FG_THR).float()
            n = fg.sum((1, 2, 3), keepdim=True).clamp(min=1)
            m = (z * fg).sum((1, 2, 3), keepdim=True) / n
            var = (((z - m) * fg) ** 2).sum(
                (1, 2, 3), keepdim=True
            ) / n
            z = (z - m) / var.sqrt().clamp(min=1e-6)

        if self.pre:
            z = self.pre(z)

        out = self.model(z)

        if self.post:
            out = self.post(out)

        p = (
            out.softmax(1)
            if out.shape[1] > 1
            else torch.cat([1 - out.sigmoid(), out.sigmoid()], 1)
        )

        # Restore the original image dimensions.
        if p.shape[-2:] != (H, W):
            p = F.interpolate(
                p,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return p


class IntensityTissue:
    """SANITY BASELINE, not a real segmenter: background / CSF / grey / white matter by clustering brain-pixel
    brightness (dark = CSF, mid = grey, bright = white on T1), fitted on real training slices. It lets you run and
    check this whole module before a proper segmenter is plugged in."""
    def __init__(self, ref_u8, iters=25):
        x = ref_u8[torch.randperm(len(ref_u8))[:500]].float().flatten() / 255
        x = x[x > FG_THR]
        x = x[torch.randperm(len(x))[:400_000]]
        c = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9]))
        for _ in range(iters):
            a = (x[:, None] - c[None]).abs().argmin(1)
            c = torch.stack([x[a == k].mean() if (a == k).any() else c[k] for k in range(3)])
        c = c.sort().values
        a = (x[:, None] - c[None]).abs().argmin(1)
        s = torch.stack([x[a == k].std().clamp(min=1e-3) for k in range(3)])
        self.c, self.s = c.to(DEVICE).view(1, 3, 1, 1), s.to(DEVICE).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        z = F.avg_pool2d(F.pad((x + 1) / 2, (1, 1, 1, 1), mode="replicate"), 3, 1)     # mild smoothing
        p = (-0.5 * ((z - self.c) / self.s) ** 2 - self.s.log()).softmax(1)
        bg = (z < FG_THR).float()
        return torch.cat([bg, p * (1 - bg)], 1)

# --------------------------------------------------------------------------- #
# --------- Construct UNet Segmenter pre-trained from OASIS dataset --------- #
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2 -- the standard U-Net building block.
    BatchNorm is not in the original 2015 paper but materially stabilises
    training at this learning rate."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Symmetric encoder-decoder. The encoder halves spatial size / doubles
    channels four times; the decoder mirrors it. Each decoder stage
    concatenates the same-resolution encoder feature map (the skip
    connection) before its DoubleConv.

    Output: (B, NUM_CLASSES, H, W) raw logits -- one channel per class, i.e.
    a categorical prediction per pixel. Softmax over dim=1 gives a per-pixel
    probability distribution; argmax gives the predicted label map.
    """

    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8, base * 16]   # 32,64,128,256,512

        # Encoder
        self.enc1 = DoubleConv(in_ch, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f[3], f[4])

        # Decoder: transposed conv upsamples, then concat skip, then DoubleConv.
        # in_ch of each DoubleConv is f[i] (upsampled) + f[i] (skip) = 2*f[i].
        self.up4  = nn.ConvTranspose2d(f[4], f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3] * 2, f[3])
        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2] * 2, f[2])
        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1] * 2, f[1])
        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0] * 2, f[0])

        # 1x1 conv projects to one logit channel per class
        self.head = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)                 # (B, 32,  256, 256)
        s2 = self.enc2(self.pool(s1))     # (B, 64,  128, 128)
        s3 = self.enc3(self.pool(s2))     # (B, 128,  64,  64)
        s4 = self.enc4(self.pool(s3))     # (B, 256,  32,  32)
        b  = self.bottleneck(self.pool(s4))  # (B, 512, 16, 16)

        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.head(d1)              # (B, NUM_CLASSES, 256, 256) logits

# DARTSeg classes excluded by its original prediction pipeline.
DART_EXCLUDED = {
    34, 35, 42, 43, 73,
    74, 77, 78, 108, 109,
}

DART_KEEP = [
    i for i in range(113)
    if i not in DART_EXCLUDED
]


def dart_post(logits):
    # [B, 113, H, W] -> [B, 103, H, W]
    return logits[:, DART_KEEP, :, :]


# dart = Segmentation(
#     model_wts_path=RUN_DIR / "dense_unet_left2right_finetuned.pth",
#     model_type="dense-unet",
#     use_gpu=torch.device(DEVICE).type == "cuda",
# )

# Wrap the actual neural network, not the Segmentation object.
# SEG = TorchSegmenter(
#     dart.model,
#     size=256,
#     norm="minmax",
#     post=dart_post,
# )

unet_model = UNet().to(DEVICE)
unet_model.load_state_dict(torch.load(RUN_DIR / "unet_mri_best.pt", map_location=DEVICE))
unet_model.eval()

SEG = None

# SEG = TorchSegmenter(unet_model)            # <- plug in your own, e.g.  SEG = TorchSegmenter(my_model, size=256, norm="zscore")
if SEG is None:
    print("No segmenter supplied: using the intensity-clustering BASELINE (CSF / grey / white by brightness). "
          "Fine for checking the pipeline, a weak test of realism.")
    SEG = IntensityTissue(train_ds.data)

# quick visual check of the segmenter on real held-out slices: do the masks look right?
chk = (val_ds if val_ds is not None else train_ds).data[torch.randperm(len((val_ds if val_ds is not None else train_ds).data))[:6]]
with torch.no_grad():
    chk_mask = SEG(as_pm1(chk)).argmax(1).cpu().numpy()
labels, counts = np.unique(chk_mask, return_counts=True)

print({
    CLASS_NAMES[int(label)]: int(count)
    for label, count in zip(labels, counts)
})
from collections import Counter

for i, mask in enumerate(chk_mask):
    counts = Counter(mask.ravel().tolist())
    total = mask.size

    print(f"\nSlice {i}:")
    for c, n in sorted(counts.items(), key=lambda item: -item[1]):
        print(
            f"  {c:2d} ({CLASS_NAMES[c]:16s}): "
            f"{n:7d} pixels ({100*n/total:5.1f}%)"
        )
fig, ax = plt.subplots(2, 6, figsize=(14, 5))
for i in range(6):
    ax[0, i].imshow(chk[i, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    ax[1, i].imshow(chk_mask[i], cmap="viridis", vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax[0, i].axis("off"); ax[1, i].axis("off")
fig.suptitle("Segmenter check on REAL held-out slices (classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)) + ")")
plt.tight_layout(); plt.show()

# ---- segment real and synthetic images, then compare (metrics 1-3) ---------------------------------------
N_REF, N_VAL, N_SYN = (300, 100, 200) if QUICK_TEST else (20, 10, 10)


@torch.no_grad()
def run_segmenter(x, bs=128):
    """x: uint8 or float [-1,1] tensor [N,1,H,W]. Returns masks, confidence (mean top-class probability over brain
    pixels) and stability (Dice between the mask of an image and the mask of a slightly shifted copy of it)."""
    masks, conf, cons = [], [], []
    for i in range(0, len(x), bs):
        b = as_pm1(x[i:i + bs])
        p = SEG(b)
        m = p.argmax(1)
        fg = (b[:, 0] > 2 * FG_THR - 1).float()
        conf.append(((p.max(1).values * fg).sum((1, 2)) / fg.sum((1, 2)).clamp(min=1)).cpu().numpy())
        m_shift = torch.roll(SEG(torch.roll(b, (3, 2), (2, 3))).argmax(1), (-3, -2), (1, 2))
        cons.append(macro_dice(dice_per_class(m.cpu().numpy(), m_shift.cpu().numpy(), TISSUE)))
        masks.append(m.cpu().numpy().astype(np.uint8))
    return dict(mask=np.concatenate(masks), conf=np.concatenate(conf), cons=np.concatenate(cons))


g_ev = torch.Generator().manual_seed(5)
ref_idx = torch.randperm(len(train_ds), generator=g_ev)[:N_REF]
x_ref = train_ds.data[ref_idx]                                          # real, TRAINING patients
x_syn = generate(N_SYN, psi=1.0, seed=123)                              # synthetic (fresh seed)
if val_ds is not None:
    x_val = val_ds.data[torch.randperm(len(val_ds), generator=g_ev)[:N_VAL]]   # real, HELD-OUT patients

print("Segmenting ...")
S_ref, S_syn = run_segmenter(x_ref), run_segmenter(x_syn)
S_val = run_segmenter(x_val) if val_ds is not None else None
low = lambda m: m[:, ::max(1, m.shape[1] // 64), ::max(1, m.shape[2] // 64)]          # 64x64 masks for the Dice matrices

# 1. class-area statistics ---------------------------------------------------------------------------------
sh = {k: tissue_share(v["mask"]) for k, v in (("ref", S_ref), ("syn", S_syn)) + ((("val", S_val),) if S_val else ())}
ks_syn = [sps.ks_2samp(sh["syn"][:, j], sh["ref"][:, j]).statistic for j in range(len(TISSUE))]
ks_val = [sps.ks_2samp(sh["val"][:, j], sh["ref"][:, j]).statistic for j in range(len(TISSUE))] if S_val else None
print("\n1. Share of the brain taken by each tissue (mean +- std)")
for j, c in enumerate(TISSUE):
    print(f"   {CLASS_NAMES[c]:5s} real {sh['ref'][:, j].mean():.3f}+-{sh['ref'][:, j].std():.3f}"
          + (f" | held-out real {sh['val'][:, j].mean():.3f}+-{sh['val'][:, j].std():.3f}" if S_val else "")
          + f" | synthetic {sh['syn'][:, j].mean():.3f}+-{sh['syn'][:, j].std():.3f}"
          + f" | KS vs real: synthetic {ks_syn[j]:.3f}" + (f", held-out {ks_val[j]:.3f}" if S_val else ""))

# 2. confidence and stability --------------------------------------------------------------------------------
print("\n2. Segmenter confidence / stability (higher = more certain / more stable)")
for name, S in (("real", S_ref),) + ((("held-out real", S_val),) if S_val else ()) + (("synthetic", S_syn),):
    print(f"   {name:14s} confidence {np.nanmedian(S['conf']):.3f} | shift-stability Dice {np.nanmean(S['cons']):.3f}")

# 3. best-match Dice against real masks ---------------------------------------------------------------------
R_low = low(S_ref["mask"])
bm_syn, i_syn = best_match_dice(low(S_syn["mask"]), R_low, TISSUE)
bm_val = best_match_dice(low(S_val["mask"]), R_low, TISSUE)[0] if S_val else None
print("\n3. Best-match Dice: how well the closest real training mask matches (mean)")
print(f"   synthetic masks {bm_syn.mean():.3f}" + (f" | held-out real masks {bm_val.mean():.3f}  (ratio {bm_syn.mean() / bm_val.mean():.2f})" if S_val else ""))

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
ax[0, 0].hist(bm_syn, 40, alpha=.6, density=True, label="synthetic")
if S_val:
    ax[0, 0].hist(bm_val, 40, alpha=.5, density=True, label="held-out real")
ax[0, 0].set_title("best-match Dice vs real training masks"); ax[0, 0].legend()
for name, S in (("real", S_ref),) + ((("held-out real", S_val),) if S_val else ()) + (("synthetic", S_syn),):
    ax[0, 1].hist(S["conf"], 40, alpha=.5, density=True, label=name)
    ax[0, 2].hist(S["cons"][~np.isnan(S["cons"])], 40, alpha=.5, density=True, label=name)
ax[0, 1].set_title("segmenter confidence"); ax[0, 1].legend(); ax[0, 2].set_title("shift-stability Dice"); ax[0, 2].legend()
for j, c in enumerate(TISSUE):
    ax[1, j].hist(sh["ref"][:, j], 40, alpha=.5, density=True, label="real")
    ax[1, j].hist(sh["syn"][:, j], 40, alpha=.5, density=True, label="synthetic")
    ax[1, j].set_title(f"share of brain: {CLASS_NAMES[c]}"); ax[1, j].legend()
plt.tight_layout(); plt.savefig(AUDIT_DIR / "segmentation_stats.png", dpi=110); plt.show()

# look at the synthetic masks and the closest real mask
n_show = 6
fig, ax = plt.subplots(4, n_show, figsize=(2.2 * n_show, 9))
for j in range(n_show):
    ax[0, j].imshow(((x_syn[j, 0] + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
    ax[1, j].imshow(S_syn["mask"][j], cmap="viridis", vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax[2, j].imshow(S_ref["mask"][i_syn[j]], cmap="viridis", vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax[3, j].imshow(x_ref[i_syn[j], 0].numpy(), cmap="gray", vmin=0, vmax=255)
    ax[1, j].set_title(f"Dice {bm_syn[j]:.2f}", fontsize=8)
    for r in range(4):
        ax[r, j].axis("off")
for r, t in enumerate(["synthetic", "its mask", "closest real mask", "that real slice"]):
    ax[r, 0].text(-0.05, 0.5, t, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(AUDIT_DIR / "segmentation_examples.png", dpi=110, bbox_inches="tight"); plt.show()

# ---- 4. paired Dice: a real scan vs the generator's reconstruction of it, then the verdict --------------------
# Both masks describe the SAME anatomy here, so this is a true real-vs-generated Dice. It tests whether G can
# represent real anatomy (training scans AND unseen held-out scans), not whether random samples are realistic.
K_PAIR = 8 if QUICK_TEST else 32
tr_i = torch.randperm(len(train_ds), generator=g_ev)[:K_PAIR]
va_i = torch.randperm(len(val_ds), generator=g_ev)[:K_PAIR] if val_ds is not None else torch.tensor([], dtype=torch.long)
pair_u8 = torch.cat([train_ds.data[tr_i]] + ([val_ds.data[va_i]] if val_ds is not None else []))
n_tr = len(tr_i)

_, recon, _ = project(as_pm1(pair_u8))                                   # [N,1,H,W] in [-1,1], CPU
m_real, m_rec = run_segmenter(pair_u8)["mask"], run_segmenter(recon)["mask"]
d_pair = macro_dice(dice_per_class(m_real, m_rec, TISSUE))
d_chance = macro_dice(dice_per_class(m_real, np.roll(m_real, 1, 0), TISSUE))             # a different slice
dice_train = float(np.nanmean(d_pair[:n_tr]))
dice_val = float(np.nanmean(d_pair[n_tr:])) if val_ds is not None else None
print(f"4. Paired Dice, real scan vs reconstruction: training scans {dice_train:.3f}"
      + (f" | held-out scans {dice_val:.3f}" if dice_val is not None else "")
      + f" | chance level (real vs a different real slice) {np.nanmean(d_chance):.3f}")
per_class = np.nanmean(dice_per_class(m_real, m_rec, TISSUE), 0)
print("   per class:", {CLASS_NAMES[c]: round(float(v), 3) for c, v in zip(TISSUE, per_class)})

fig, ax = plt.subplots(4, 4, figsize=(9, 9))
rows = list(range(min(2, n_tr))) + (list(range(n_tr, n_tr + min(2, len(pair_u8) - n_tr))) if val_ds is not None else [])
for r, k in zip(range(4), rows):
    ax[r, 0].imshow(pair_u8[k, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    ax[r, 1].imshow(m_real[k], cmap="viridis", vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax[r, 2].imshow(((recon[k, 0] + 1) / 2).numpy(), cmap="gray", vmin=0, vmax=1)
    ax[r, 3].imshow(m_rec[k], cmap="viridis", vmin=0, vmax=len(CLASS_NAMES) - 1)
    ax[r, 3].set_title(f"Dice {d_pair[k]:.2f}  ({'train' if k < n_tr else 'held-out'})", fontsize=8)
    for c in range(4):
        ax[r, c].axis("off")
for c, t in enumerate(["real", "real mask", "reconstruction", "its mask"]):
    ax[0, c].set_title(t, fontsize=9) if c < 3 else None
plt.tight_layout(); plt.savefig(AUDIT_DIR / "paired_dice.png", dpi=110); plt.show()

# ---- verdict ------------------------------------------------------------------------------------------------
seg_checks = []

def grade_seg(name, value, ok, warn, higher_is_better=False, fmt="{:.3f}"):
    if value is None:
        seg_checks.append((name, "n/a", "SKIP", "needs held-out patients"))
        return
    good, fine = (value >= ok, value >= warn) if higher_is_better else (value <= ok, value <= warn)
    rule = f"PASS {'>=' if higher_is_better else '<='} {ok}, WARN {'>=' if higher_is_better else '<='} {warn}"
    seg_checks.append((name, fmt.format(value), "PASS" if good else "WARN" if fine else "FAIL", rule))

have_val = S_val is not None
grade_seg("tissue-share KS: worst excess over real-vs-real", max(a - b for a, b in zip(ks_syn, ks_val)) if have_val else None, 0.10, 0.25)
grade_seg("confidence drop (held-out real - synthetic)", float(np.nanmedian(S_val["conf"]) - np.nanmedian(S_syn["conf"])) if have_val else None, 0.03, 0.08)
grade_seg("stability drop (held-out real - synthetic)", float(np.nanmean(S_val["cons"]) - np.nanmean(S_syn["cons"])) if have_val else None, 0.03, 0.08)
grade_seg("best-match Dice ratio (synthetic / held-out real)", float(bm_syn.mean() / bm_val.mean()) if have_val else None, 0.95, 0.85, higher_is_better=True, fmt="{:.2f}")
grade_seg("paired Dice gap (training - held-out scans)", None if dice_val is None else dice_train - dice_val, 0.03, 0.08)

# print("\n" + "=" * 100)
# print(f"{'check':58s}{'value':>9s}  status  rule")
# for nm, val, st, rule in seg_checks:
#     print(f"{nm:58s}{val:>9s}  {st:6s}  {rule}")
# sts = [c[2] for c in seg_checks if c[2] != "SKIP"]
# overall = "NO DATA" if not sts else "FAIL" if "FAIL" in sts else "WARN" if "WARN" in sts else "PASS"
# TEXT = {"PASS": "the segmenter treats synthetic slices like real ones: similar tissue proportions, confidence and stability, and real masks of comparable fit exist for them.",
#         "WARN": "partly consistent: at least one measure separates synthetic from real. Compare segmentation_examples.png and the histograms.",
#         "FAIL": "the segmenter behaves clearly differently on synthetic slices (unusual tissue proportions, low confidence or stability, or masks no real slice resembles).",
#         "NO DATA": "no check could be computed (no held-out patients)."}
# print(f"\nVERDICT (segmentation-based): {overall}: {TEXT[overall]}")
# print(f"  Segmenter used: {type(SEG).__name__}" + ("  <- weak baseline, replace with a real segmenter for a meaningful result" if isinstance(SEG, IntensityTissue) else ""))
# print("  Caveat: a segmenter can produce tidy-looking masks on unrealistic images, so PASS is not proof of realism; a FAIL is informative.")
# print("          The thresholds are rules of thumb. Judge the figures in", AUDIT_DIR)