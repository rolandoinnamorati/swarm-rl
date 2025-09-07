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

class Continuous2DEnvMoving:
    def __init__(self, L=100.0, dt=0.1, v_max=10.0, a_max=5.0, max_steps=400, eps=1.0,
                 include_angle=True, include_time=True, include_tgt_vel=True,
                 tgt_speed=0.5, seed=None):
        self.L = float(L); self.dt = float(dt)
        self.v_max = float(v_max); self.a_max = float(a_max)
        self.max_steps = int(max_steps); self.eps = float(eps)
        self.include_angle = include_angle; self.include_time = include_time
        self.include_tgt_vel = include_tgt_vel
        self.tgt_speed = float(tgt_speed)
        self.rng = np.random.default_rng(seed)
        self.diag_max = math.sqrt(2.0) * self.L
        self.p = None; self.v = None
        self.pt = None; self.vt = None
        self.t = 0
    def _spawn_positions(self):
        p = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        pt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        while np.linalg.norm(pt - p) < 1e-3:
            pt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        return p, pt
    def _spawn_target_velocity(self):
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        return np.array([math.cos(theta)*self.tgt_speed, math.sin(theta)*self.tgt_speed], dtype=np.float32)
    def _bounce(self, pos, vel):
        x, y = pos; vx, vy = vel
        if x < 0.0: x = 0.0; vx = abs(vx)
        elif x > self.L: x = self.L; vx = -abs(vx)
        if y < 0.0: y = 0.0; vy = abs(vy)
        elif y > self.L: y = self.L; vy = -abs(vy)
        return np.array([x, y], dtype=np.float32), np.array([vx, vy], dtype=np.float32)
    def reset(self):
        self.p, self.pt = self._spawn_positions(); self.v = np.zeros(2, dtype=np.float32)
        self.vt = self._spawn_target_velocity(); self.t = 0
        d = float(np.linalg.norm(self.pt - self.p))
        return self._obs(), {"dist": d}
    def _obs(self):
        dp = (self.pt - self.p) / max(1e-6, self.L)
        v_norm = self.v / max(1e-6, self.v_max)
        feats = [dp[0], dp[1], v_norm[0], v_norm[1]]
        if self.include_tgt_vel:
            vt_norm = self.vt / max(1e-6, self.v_max)
            feats += [vt_norm[0], vt_norm[1]]
        if self.include_angle:
            v = self.v; dp_raw = self.pt - self.p
            nv = np.linalg.norm(v) + 1e-8; ndp = np.linalg.norm(dp_raw) + 1e-8
            cos_th = float(np.clip(np.dot(v, dp_raw) / (nv * ndp), -1.0, 1.0))
            cross = float(v[0]*dp_raw[1] - v[1]*dp_raw[0])
            sin_th = float(np.clip(cross / (nv * ndp), -1.0, 1.0))
            feats += [cos_th, sin_th]
        if self.include_time:
            feats += [1.0 - (self.t / max(1, self.max_steps))]
        return np.asarray(feats, dtype=np.float32)
    def step(self, a):
        a = np.asarray(a, dtype=np.float32); a = np.clip(a, -self.a_max, self.a_max)
        self.v = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
        self.p = np.clip(self.p + self.v * self.dt, 0.0, self.L)
        pt_next = self.pt + self.vt * self.dt
        self.pt, self.vt = self._bounce(pt_next, self.vt)
        self.t += 1
        d = float(np.linalg.norm(self.pt - self.p))
        done = (d <= self.eps) or (self.t >= self.max_steps)
        return self._obs(), 0.0, done, {"dist": d, "success": (d <= self.eps)}

class StaticNorm:
    def __init__(self, mean, var, clip=5.0):
        self.mean = torch.tensor(mean, dtype=TORCH_DTYPE, device=DEVICE)
        self.var = torch.tensor(var, dtype=TORCH_DTYPE, device=DEVICE).clamp_min(1e-8)
        self.clip = clip
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.sqrt(self.var)
        return torch.clamp(z, -self.clip, self.clip)

def load_norm(ckpt_dir, obs_dim, include_angle, include_time, include_tgt_vel):
    path = os.path.join(ckpt_dir, "obs_stats.npz")
    if os.path.isfile(path):
        data = np.load(path)
        print("[INFO] Loaded observation stats from obs_stats.npz")
        return StaticNorm(data["mean"], data["var"])
    feats = []
    feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # dp
    feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # v
    if include_tgt_vel:
        feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # vt scaled with v_max
    if include_angle:
        feats += [(0.0, 0.5), (0.0, 0.5)]
    if include_time:
        feats += [(0.5, 1.0/12.0)]
    mean = np.array([m for (m, v) in feats], dtype=np.float32)
    var  = np.array([v for (m, v) in feats], dtype=np.float32)
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
        log_std = self.log_std.clamp(-2.5, 1.0)
        return mu, log_std

class TanhNormal:
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor):
        self.mu = mu; self.log_std = log_std; self.std = torch.exp(log_std)
    def deterministic(self):
        return torch.tanh(self.mu)
    def sample(self):
        eps = torch.randn_like(self.mu); z = self.mu + self.std * eps
        return torch.tanh(z)

def run_episode(actor: Actor, norm: StaticNorm, cfg: dict, stochastic: bool, seed=None):
    env = Continuous2DEnvMoving(**cfg, seed=seed)
    obs, info = env.reset()
    done = False
    frames = []
    dists, speeds, speeds_t = [], [], []
    t = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
        obs_t = norm.normalize(obs_t)
        mu, log_std = actor(obs_t)
        dist = TanhNormal(mu, log_std)
        a = dist.sample() if stochastic else dist.deterministic()
        a = (a * cfg["a_max"]).squeeze(0).cpu().numpy()
        obs, _, done, info = env.step(a)
        frames.append((env.p.copy(), env.pt.copy()))
        dists.append(info["dist"])
        speeds.append(float(np.linalg.norm(env.v)))
        speeds_t.append(float(np.linalg.norm(env.vt)))
        t += 1
        if t >= cfg["max_steps"]:
            break
    summary = {"success": bool(info.get("success", False)), "steps": t,
               "final_dist": float(dists[-1]) if dists else float("nan"),
               "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
               "avg_speed_t": float(np.mean(speeds_t)) if speeds_t else 0.0}
    return frames, dists, speeds, speeds_t, summary, cfg["L"]


def animate(frames, L, dists, speeds, speeds_t, out_path, fps=20, trail=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_title("P3 PPO — Moving Target (CV)")

    drone_pt, = ax.plot([], [], 'o', markersize=8, label='drone')
    tgt_pt,   = ax.plot([], [], 'X', markersize=10, label='target')
    path_ln,  = ax.plot([], [], '-', linewidth=2, alpha=0.8, label='drone path')
    tgt_ln,   = ax.plot([], [], ':', linewidth=1.5, alpha=0.7, label='target path')
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')

    xs, ys, xts, yts = [], [], [], []

    def init():
        for ln in (path_ln, tgt_ln): ln.set_data([], [])
        for pt in (drone_pt, tgt_pt): pt.set_data([], [])
        txt.set_text("")
        return drone_pt, tgt_pt, path_ln, tgt_ln, txt

    def update(i):
        p, pt = frames[i]
        xs.append(p[0]); ys.append(p[1])
        xts.append(pt[0]); yts.append(pt[1])
        drone_pt.set_data([p[0]], [p[1]])
        tgt_pt.set_data([pt[0]], [pt[1]])
        if trail:
            path_ln.set_data(xs, ys)
            tgt_ln.set_data(xts, yts)
        txt.set_text(f"t={i}  dist={dists[i]:.2f}  |v|={speeds[i]:.2f}  |vt|={speeds_t[i]:.2f}")
        return drone_pt, tgt_pt, path_ln, tgt_ln, txt

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=int(1000/max(1,fps)), blit=False)

    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext == '.mp4':
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(out_path, writer=writer)
        elif ext == '.gif':
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
        else:
            raise ValueError('use .mp4 or .gif')
        print(f"[OK] saved animation: {out_path}")
    finally:
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', default='checkpoints_p3')
    ap.add_argument('--out', default='episode_p3.mp4')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()

    cfg_path = os.path.join(args.ckpt_dir, 'config_p3.npz')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    z = np.load(cfg_path, allow_pickle=True)
    env_cfg = dict(
        L=float(z['L']), dt=float(z['dt']), v_max=float(z['v_max']), a_max=float(z['a_max']),
        max_steps=int(z['max_steps']), eps=float(z['eps']),
        include_angle=bool(z['include_angle']), include_time=bool(z['include_time']),
        include_tgt_vel=bool(z['include_tgt_vel']), tgt_speed=float(z['tgt_speed']),
    )
    obs_dim = int(z['obs_dim']); act_dim = int(z['act_dim'])

    # policy
    actor = Actor(obs_dim, act_dim).to(DEVICE)
    w_path = os.path.join(args.ckpt_dir, 'ppo_actor.pt')
    if not os.path.isfile(w_path):
        raise FileNotFoundError(f"Actor not found: {w_path}")
    actor.load_state_dict(torch.load(w_path, map_location=DEVICE))
    actor.eval()

    # normalization
    stats_path = os.path.join(args.ckpt_dir, 'obs_stats.npz')
    if os.path.isfile(stats_path):
        data = np.load(stats_path)
        print('[INFO] Loaded observation stats')
        mean = data['mean']; var = data['var']
    else:
        feats = []
        feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # dp
        feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # v
        if env_cfg['include_tgt_vel']:
            feats += [(0.0, 1.0/3.0), (0.0, 1.0/3.0)]  # vt
        if env_cfg['include_angle']:
            feats += [(0.0, 0.5), (0.0, 0.5)]
        if env_cfg['include_time']:
            feats += [(0.5, 1.0/12.0)]
        mean = np.array([m for (m, v) in feats], dtype=np.float32)
        var  = np.array([v for (m, v) in feats], dtype=np.float32)

    class StaticNorm:
        def __init__(self, mean, var, clip=5.0):
            self.mean = torch.tensor(mean, dtype=TORCH_DTYPE, device=DEVICE)
            self.var = torch.tensor(var, dtype=TORCH_DTYPE, device=DEVICE).clamp_min(1e-8)
            self.clip = clip
        def normalize(self, x):
            z = (x - self.mean) / torch.sqrt(self.var)
            return torch.clamp(z, -self.clip, self.clip)

    norm = StaticNorm(mean, var)
    env_cfg['tgt_speed'] = 9.0

    print(f"[PLAY P3] L={env_cfg['L']} dt={env_cfg['dt']} v_max={env_cfg['v_max']} a_max={env_cfg['a_max']} "
          f"max_steps={env_cfg['max_steps']} eps={env_cfg['eps']} tgt_speed={env_cfg['tgt_speed']} obs_dim={obs_dim} act_dim={act_dim} stochastic={args.stochastic}")

    frames, dists, speeds, speeds_t, summary, L = run_episode(actor, norm, env_cfg, args.stochastic, seed=args.seed)
    print(f"[EP] success={summary['success']} steps={summary['steps']} final_dist={summary['final_dist']:.2f} "
          f"avg|v|={summary['avg_speed']:.2f} avg|vt|={summary['avg_speed_t']:.2f}")

    animate(frames, L, dists, speeds, speeds_t, args.out, fps=args.fps, trail=True)

if __name__ == '__main__':
    main()