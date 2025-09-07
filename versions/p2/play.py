import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32
torch.set_grad_enabled(False)

class Continuous2DEnv:
    def __init__(self, L=100.0, dt=0.1, v_max=10.0, a_max=5.0, max_steps=400, eps=1.0,
                 include_angle=True, include_time=True, seed=None):
        self.L = float(L)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.include_angle = include_angle
        self.include_time = include_time
        self.rng = np.random.default_rng(seed)
        self.diag_max = math.sqrt(2.0) * self.L

        self.p = None  # (x,y)
        self.v = None  # (vx,vy)
        self.tgt = None
        self.t = 0

    def _spawn(self):
        p = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        tgt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        while np.allclose(p, tgt, atol=1e-3):
            tgt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        return p, tgt

    def reset(self):
        self.p, self.tgt = self._spawn()
        self.v = np.zeros(2, dtype=np.float32)
        self.t = 0
        d = float(np.linalg.norm(self.tgt - self.p))
        return self._obs(d), {"dist": d}

    def _obs(self, d_curr=None):
        dp = (self.tgt - self.p) / max(1e-6, self.L)        # [-1,1] approx
        v_norm = self.v / max(1e-6, self.v_max)             # [-1,1] approx
        feats = [dp[0], dp[1], v_norm[0], v_norm[1]]

        if self.include_angle:
            v = self.v
            dp_raw = self.tgt - self.p
            nv = np.linalg.norm(v) + 1e-8
            ndp = np.linalg.norm(dp_raw) + 1e-8
            cos_th = float(np.clip(np.dot(v, dp_raw) / (nv * ndp), -1.0, 1.0))
            cross = float(v[0]*dp_raw[1] - v[1]*dp_raw[0])
            sin_th = float(np.clip(cross / (nv * ndp), -1.0, 1.0))
            feats += [cos_th, sin_th]

        if self.include_time:
            feats += [1.0 - (self.t / max(1, self.max_steps))]

        return np.asarray(feats, dtype=np.float32)

    def step(self, a):
        # a: np.array shape (2,), continuous accel command (m/s^2), clip to a_max
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -self.a_max, self.a_max)

        self.v = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
        self.p = np.clip(self.p + self.v * self.dt, 0.0, self.L)
        self.t += 1

        d = float(np.linalg.norm(self.tgt - self.p))
        done = (d <= self.eps) or (self.t >= self.max_steps)
        info = {"dist": d, "success": (d <= self.eps)}
        return self._obs(d), 0.0, done, info  # reward non usato in play

class StaticNorm:
    def __init__(self, mean: np.ndarray, var: np.ndarray, clip: float = 5.0):
        self.mean = torch.tensor(mean, dtype=TORCH_DTYPE, device=DEVICE)
        self.var = torch.tensor(var, dtype=TORCH_DTYPE, device=DEVICE).clamp_min(1e-8)
        self.clip = clip

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.sqrt(self.var)
        return torch.clamp(z, -self.clip, self.clip)

def load_norm_or_fallback(ckpt_dir: str, obs_dim: int, include_angle: bool, include_time: bool) -> StaticNorm:
    path = os.path.join(ckpt_dir, "obs_stats.npz")
    if os.path.isfile(path):
        data = np.load(path)
        mean = data["mean"]
        var = data["var"]
        if mean.shape[0] != obs_dim or var.shape[0] != obs_dim:
            print(f"[WARN] obs_stats.npz shape mismatch (found {mean.shape}, expected {obs_dim}); using theory fallback.")
        else:
            print("[INFO] Loaded observation stats from obs_stats.npz")
            return StaticNorm(mean, var)

    feats = []
    # dp_x, dp_y
    feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]
    # v_x, v_y
    feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]
    if include_angle:
        feats += [(0.0, 0.5), (0.0, 0.5)]
    if include_time:
        feats += [(0.5, 1.0/12.0)]
    mean = np.array([m for (m, v) in feats], dtype=np.float32)
    var  = np.array([v for (m, v) in feats], dtype=np.float32)
    print("[WARN] Using theoretical fallback for observation normalization (save obs_stats.npz for exact parity).")
    return StaticNorm(mean, var)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor):
        mu = self.net(obs)
        log_std = self.log_std.clamp(-3.0, 2.0)
        return mu, log_std

class TanhNormal:
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor):
        self.mu = mu
        self.log_std = log_std
        self.std = torch.exp(log_std)

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.mu)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        z = self.mu + self.std * eps
        return torch.tanh(z)

def run_episode(actor: Actor, norm: StaticNorm, cfg: dict, stochastic: bool, seed=None):
    env = Continuous2DEnv(**cfg, seed=seed)
    obs, info = env.reset()
    done = False

    frames = []
    dists, speeds, accels = [], [], []
    actions = []

    t = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
        obs_t = norm.normalize(obs_t)
        mu, log_std = actor(obs_t)
        dist = TanhNormal(mu, log_std)
        if stochastic:
            a = dist.sample()
        else:
            a = dist.deterministic()
        a = (a * cfg["a_max"]).squeeze(0).cpu().numpy()

        # Step
        obs, _, done, info = env.step(a)

        # Logs & frames
        frames.append((env.p.copy(), env.tgt.copy()))
        dists.append(info["dist"])
        speeds.append(float(np.linalg.norm(env.v)))
        accels.append(float(np.linalg.norm(a)))
        actions.append(a.copy())

        t += 1
        if t >= cfg["max_steps"]:
            break

    summary = {
        "success": bool(info.get("success", False)),
        "steps": t,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "avg_accel": float(np.mean(accels)) if accels else 0.0,
        "final_dist": float(dists[-1]) if dists else float("nan"),
    }
    return frames, dists, speeds, actions, summary, cfg["L"]

def animate(frames, L, dists, speeds, out_path, fps=20, trail=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("P2a PPO — 2D Continuous Interception")

    drone_pt, = ax.plot([], [], 'o', markersize=8, label="drone")
    tgt_pt,   = ax.plot([], [], 'X', markersize=10, label="target")
    path_ln,  = ax.plot([], [], '-', linewidth=2, alpha=0.8, label="path")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    xs, ys = [], []

    def init():
        drone_pt.set_data([], [])
        tgt_pt.set_data([], [])
        path_ln.set_data([], [])
        txt.set_text("")
        return drone_pt, tgt_pt, path_ln, txt

    def update(i):
        (p, tgt) = frames[i]
        xs.append(p[0]); ys.append(p[1])
        drone_pt.set_data([p[0]], [p[1]])
        tgt_pt.set_data([tgt[0]], [tgt[1]])
        if trail:
            path_ln.set_data(xs, ys)
        msg = f"t={i}   dist={dists[i]:.2f}   |v|={speeds[i]:.2f}"
        txt.set_text(msg)
        return drone_pt, tgt_pt, path_ln, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames),
                                   interval=int(1000/max(1, fps)), blit=False)

    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(out_path, writer=writer)
        elif ext == ".gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
        else:
            raise ValueError("Usa estensione .mp4 o .gif")
        print(f"[OK] Animazione salvata: {out_path}")
    finally:
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints_cont", help="cartella con ppo_actor.pt e config_continuous.npz")
    ap.add_argument("--out", default="episode_cont.mp4", help="output .mp4 o .gif")
    ap.add_argument("--seed", type=int, default=None, help="seed per lo spawn; None=random")
    ap.add_argument("--stochastic", action="store_true", help="usa campionamento (default: deterministico)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no_trail", action="store_true", help="disabilita la scia del percorso")
    ap.add_argument("--norm", choices=["auto","identity","theory"], default="auto",
                    help="auto: carica obs_stats.npz o fallback teorico; identity: nessuna normalizzazione; theory: sempre fallback teorico")
    args = ap.parse_args()

    # Load config
    cfg_path = os.path.join(args.ckpt_dir, "config_continuous.npz")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config non trovata: {cfg_path}")
    cfg_npz = np.load(cfg_path, allow_pickle=True)
    env_cfg = dict(
        L=float(cfg_npz["L"]),
        dt=float(cfg_npz["dt"]),
        v_max=float(cfg_npz["v_max"]),
        a_max=float(cfg_npz["a_max"]),
        max_steps=int(cfg_npz["max_steps"]),
        eps=float(cfg_npz["eps"]),
        include_angle=bool(cfg_npz["include_angle"]),
        include_time=bool(cfg_npz["include_time"]),
    )
    obs_dim = int(cfg_npz["obs_dim"])
    act_dim = int(cfg_npz["act_dim"])

    # Build actor and load weights
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    w_path = os.path.join(args.ckpt_dir, "ppo_actor.pt")
    if not os.path.isfile(w_path):
        raise FileNotFoundError(f"Actor non trovato: {w_path}")
    actor.load_state_dict(torch.load(w_path, map_location=DEVICE))
    actor.eval()

    # Observation normalization
    if args.norm == "identity":
        mean = np.zeros(obs_dim, dtype=np.float32); var = np.ones(obs_dim, dtype=np.float32)
        norm = StaticNorm(mean, var)
        print("[WARN] Using identity normalization (no z-score).")
    elif args.norm == "theory":
        norm = load_norm_or_fallback(args.ckpt_dir, obs_dim, env_cfg["include_angle"], env_cfg["include_time"])
    else:  # auto
        norm = load_norm_or_fallback(args.ckpt_dir, obs_dim, env_cfg["include_angle"], env_cfg["include_time"])

    print(f"[PLAY] L={env_cfg['L']} dt={env_cfg['dt']} v_max={env_cfg['v_max']} a_max={env_cfg['a_max']} "
          f"max_steps={env_cfg['max_steps']} eps={env_cfg['eps']} obs_dim={obs_dim} act_dim={act_dim} "
          f"stochastic={args.stochastic}")

    frames, dists, speeds, actions, summary, L = run_episode(actor, norm, env_cfg, args.stochastic, seed=args.seed)
    print(f"[EP] success={summary['success']} steps={summary['steps']} "
          f"final_dist={summary['final_dist']:.2f} avg|v|={summary['avg_speed']:.2f} avg|a|={summary['avg_accel']:.2f}")
    animate(frames, L, dists, speeds, args.out, fps=args.fps, trail=(not args.no_trail))

if __name__ == "__main__":
    main()
