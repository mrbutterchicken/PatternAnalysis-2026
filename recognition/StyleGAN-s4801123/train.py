"""
Contains the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be 
imported from dataset.py. 
Make sure to plot the losses and metrics during training.
"""
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon (M1/M2/M3/M4)
else:
    device = torch.device("cpu")        # Fallback Default

print(f"Using device: {device}")

from dataset import *
from modules import *

# =============================================================================
# TRAINING
# =============================================================================
class Meter:
    """Collects scalar tensors every iteration; averages them at log time (one GPU sync per log line)."""
    def __init__(self):
        self.v = defaultdict(list)

    def add(self, k, t):
        self.v[k].append(t.detach().float())

    def flush(self):
        out = {k: torch.stack(v).mean().item() for k, v in self.v.items()}
        self.v.clear()
        return out


def plot_log(log, path=None, show_fig=True):
    if not log:
        return
    k = [r["kimg"] for r in log]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for key in ("loss_D", "loss_G"):
        ax[0, 0].plot(k, [r[key] for r in log], label=key)
    ax[0, 0].set_title("losses"); ax[0, 0].legend()
    for key in ("D_real", "D_fake"):
        ax[0, 1].plot(k, [r[key] for r in log], label=key)
    ax[0, 1].set_title("discriminator logits"); ax[0, 1].legend()
    for key in ("std_real", "std_fake"):
        ax[1, 0].plot(k, [r[key] for r in log], label=key)
    ax[1, 0].set_title("minibatch std (per-pixel std across the batch): collapse monitor"); ax[1, 0].legend()
    ax[1, 1].plot(k, [r["ada_p"] for r in log], label="augmentation p")
    ax[1, 1].plot(k, [r["rt"] for r in log], label="r_t = mean sign D(real)")
    ax[1, 1].set_title("adaptive augmentation / D overfitting signal"); ax[1, 1].legend()
    for a in ax.flat:
        a.set_xlabel("kimg (thousands of images shown to D)")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.show() if show_fig else plt.close(fig)


# ---- models, optimisers, resume ----------------------------------------------------------------
G, D, G_ema = build_models(RES)
c = R1_EVERY / (R1_EVERY + 1)                                  # lazy-regularisation correction for D's optimiser
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.99))
opt_D = torch.optim.Adam(D.parameters(), lr=LR * c, betas=(0.0, 0.99 ** c))

start_epoch, it, ada_p, log = 0, 0, 0.0, []
if RESUME and not QUICK_TEST and (RUN_DIR / "latest.pt").exists():
    ck = torch.load(RUN_DIR / "latest.pt", map_location=DEVICE, weights_only=False)
    assert ck["res"] == RES, "checkpoint resolution differs from the dataset resolution"
    G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); G_ema.load_state_dict(ck["G_ema"])
    opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
    start_epoch, it, ada_p, log = ck["epoch"], ck["it"], ck["ada_p"], ck["log"]
    print(f"Resumed from epoch {start_epoch} (iteration {it}, augmentation p {ada_p:.3f})")


def save_ckpt(epoch):
    """latest.pt = everything needed to resume (overwritten); snapshots/ = small EMA-generator-only files, one per save."""
    state = dict(G=G.state_dict(), D=D.state_dict(), G_ema=G_ema.state_dict(), opt_G=opt_G.state_dict(),
                 opt_D=opt_D.state_dict(), epoch=epoch, it=it, ada_p=ada_p, log=log, res=RES)
    tmp = RUN_DIR / "latest.tmp"
    torch.save(state, tmp)
    os.replace(tmp, RUN_DIR / "latest.pt")                    # atomic: a disconnect mid-write can't corrupt latest.pt
    torch.save(dict(G_ema=G_ema.state_dict(), res=RES, epoch=epoch), RUN_DIR / "snapshots" / f"G_ema_epoch{epoch:04d}.pt")
    if log:
        with open(RUN_DIR / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in log for k in r}), restval="")
            w.writeheader(); w.writerows(log)


fixed_z = torch.randn(16, Z_DIM, generator=torch.Generator().manual_seed(123)).to(DEVICE)
iters_per_epoch = 20 if QUICK_TEST else max(1, len(train_ds) // BATCH)
n_epochs = 2 if QUICK_TEST else EPOCHS
save_every = 1 if QUICK_TEST else SAVE_EVERY
log_every = 10 if QUICK_TEST else LOG_EVERY
print(f"{iters_per_epoch} iterations/epoch x {n_epochs} epochs = {iters_per_epoch * n_epochs * BATCH / 1000:.0f} kimg in total")

# ---- the loop ----------------------------------------------------------------------------------
G.train(); D.train()
meter, rt_buf, last_rt, low_streak = Meter(), [], 0.0, 0
loader = batches(train_ds, BATCH)
t_log = time.time()
epoch = start_epoch - 1

try:
    for epoch in range(start_epoch, n_epochs):
        for _ in range(iters_per_epoch):
            real = next(loader)

            # ---- discriminator: non-saturating logistic loss ----
            with torch.no_grad():
                fake = G(torch.randn(BATCH, Z_DIM, device=DEVICE), mix_prob=STYLE_MIX_PROB)
            d_real, d_fake = D(augment(real, ada_p)), D(augment(fake, ada_p))
            loss_D = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
            opt_D.zero_grad(set_to_none=True); loss_D.backward(); opt_D.step()

            if it % R1_EVERY == 0:                             # lazy R1: penalise D's gradient on real images
                real_aug = augment(real, ada_p).detach().requires_grad_(True)
                grad, = torch.autograd.grad(D(real_aug).sum(), real_aug, create_graph=True)
                r1 = grad.square().sum([1, 2, 3]).mean()
                opt_D.zero_grad(set_to_none=True); (r1 * (R1_GAMMA / 2) * R1_EVERY).backward(); opt_D.step()
                meter.add("r1", r1)

            # adaptive augmentation: raise p when D is too sure about real images (overfitting), lower it otherwise
            rt_buf.append(d_real.detach().sign().mean())
            if len(rt_buf) == 4:
                last_rt = torch.stack(rt_buf).mean().item(); rt_buf.clear()
                ada_p = min(1.0, max(0.0, ada_p + math.copysign(BATCH * 4 / (ADA_KIMG * 1000), last_rt - ADA_TARGET)))

            # ---- generator ----
            D.requires_grad_(False)
            fake = G(torch.randn(BATCH, Z_DIM, device=DEVICE), mix_prob=STYLE_MIX_PROB)
            loss_G = F.softplus(-D(augment(fake, ada_p))).mean()
            opt_G.zero_grad(set_to_none=True); loss_G.backward(); opt_G.step()
            D.requires_grad_(True)

            # ---- EMA generator (what we sample from) ----
            with torch.no_grad():
                ema_nimg = min(EMA_KIMG * 1000, it * BATCH * 0.05)              # ramp-up: track fast early on
                beta = 0.5 ** (BATCH / max(ema_nimg, 1e-8))
                for pe, p in zip(G_ema.parameters(), G.parameters()):
                    pe.lerp_(p, 1 - beta)
                for be, b in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(b)

            # ---- collapse monitor: per-pixel std across every minibatch, real vs generated ----
            meter.add("std_real", real.std(0).mean())
            meter.add("std_fake", fake.detach().std(0).mean())
            for k_, v_ in (("loss_D", loss_D), ("loss_G", loss_G), ("D_real", d_real.mean()), ("D_fake", d_fake.mean())):
                meter.add(k_, v_)

            it += 1
            if it % log_every == 0:
                s = meter.flush()
                s.update(it=it, epoch=epoch + 1, kimg=it * BATCH / 1000, ada_p=ada_p, rt=last_rt,
                         img_s=log_every * BATCH / (time.time() - t_log))
                t_log = time.time()
                s["std_ratio"] = s["std_fake"] / s["std_real"]
                if not all(math.isfinite(v) for v in s.values()):
                    raise FloatingPointError("NaN/inf in the logs: stop, and resume from the last checkpoint with a lower LR")
                log.append(s)
                print(f"ep {epoch + 1:3d} it {it:6d} | {s['kimg']:7.1f} kimg | loss D {s['loss_D']:.3f} G {s['loss_G']:.3f} | "
                      f"D(real) {s['D_real']:+.2f} D(fake) {s['D_fake']:+.2f} | "
                      f"batch std real {s['std_real']:.3f} fake {s['std_fake']:.3f} ratio {s['std_ratio']:.2f} | "
                      f"aug p {ada_p:.2f} | {s['img_s']:.0f} img/s")
                low_streak = low_streak + 1 if (s["kimg"] > 50 and s["std_ratio"] < 0.6) else 0
                if low_streak >= 5:
                    print("   WARNING: generated batches are much less varied than real ones for 5 log intervals in a row "
                          "(possible mode collapse). Check samples/; consider stopping and resuming from an earlier snapshot.")

        if (epoch + 1) % save_every == 0 or epoch + 1 == n_epochs:
            save_ckpt(epoch + 1)
            with torch.no_grad():
                save_image((G_ema(fixed_z, noise_mode="zero").clamp(-1, 1) + 1) / 2,
                           RUN_DIR / "samples" / f"epoch{epoch + 1:04d}.png", nrow=4)
            plot_log(log, RUN_DIR / "curves.png", show_fig=False)
            print(f"-- saved checkpoint, samples and curves at epoch {epoch + 1} --")
except KeyboardInterrupt:
    print("Interrupted: saving a checkpoint so you can resume later.")
    save_ckpt(epoch)

plot_log(log, RUN_DIR / "curves.png")
with torch.no_grad():
    show(G_ema(fixed_z, noise_mode="zero"), nrow=8, title=f"EMA generator, fixed latents (epoch {epoch + 1})")