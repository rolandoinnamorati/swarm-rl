import os, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def load_config(ckpt_dir):
    data = np.load(os.path.join(ckpt_dir, "config.npz"), allow_pickle=True)
    plane_size = int(data["plane_size"])
    max_steps = int(data["max_steps"])
    eps = float(data["eps"])
    actions = data["actions"]
    return plane_size, max_steps, eps, actions

class ReplayEnv:
    def __init__(self, plane_size, max_steps, eps, rng):
        self.plane = plane_size
        self.max_steps = max_steps
        self.eps = eps
        self.rng = rng
        self.t = 0
        self.state = None

    def reset(self):
        d = (self.rng.integers(0, self.plane), self.rng.integers(0, self.plane))
        t = (self.rng.integers(0, self.plane), self.rng.integers(0, self.plane))
        while d == t:
            t = (self.rng.integers(0, self.plane), self.rng.integers(0, self.plane))
        self.state = (d, t)
        self.t = 0
        return self._obs()

    def _obs(self):
        (dx, dy), (tx, ty) = self.state
        denom = max(1, self.plane - 1)
        return np.array([dx, dy, tx, ty], dtype=np.float32) / denom

    def step(self, a_vec):
        (x, y), (tx, ty) = self.state
        nx = int(np.clip(x + a_vec[0], 0, self.plane - 1))
        ny = int(np.clip(y + a_vec[1], 0, self.plane - 1))
        self.t += 1
        self.state = ((nx, ny), (tx, ty))
        dist = float(np.hypot(tx - nx, ty - ny))
        done = (dist <= self.eps) or (self.t >= self.max_steps)
        return self._obs(), dist, done

    def raw_state(self):
        return self.state

def greedy_action(qnet: QNet, obs: np.ndarray):
    with torch.no_grad():
        q = qnet(torch.from_numpy(obs).to(DEVICE).unsqueeze(0))
        a = int(q.argmax(dim=1).item())
    return a

def greedy_with_fallback(qnet, obs, actions, raw_state, flat_tol=1e-6):
    with torch.no_grad():
        q = qnet(torch.from_numpy(obs).to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
    if np.ptp(q) <= flat_tol:
        (dx, dy), (tx, ty) = raw_state
        best_a, best_d = 0, float("inf")
        for a_id, (ax, ay) in enumerate(actions):
            nx = int(np.clip(dx + ax, 0, actions.shape[0]-1)) # clamp per sicurezza
            ny = int(np.clip(dy + ay, 0, actions.shape[0]-1))
            d2 = (tx - nx)**2 + (ty - ny)**2
            if d2 < best_d:
                best_d, best_a = d2, a_id
        return best_a
    return int(np.argmax(q))

def generate_episode(qnet, plane, max_steps, eps, actions, seed=None):
    rng = np.random.default_rng(seed)
    env = ReplayEnv(plane, max_steps, eps, rng)
    obs = env.reset()
    frames = [env.raw_state()]
    dists = []
    done = False
    while not done:
        a_id = greedy_with_fallback(qnet, obs, actions, env.raw_state())
        a_vec = actions[a_id]
        obs, dist, done = env.step(a_vec)
        frames.append(env.raw_state())
        dists.append(dist)
    return frames, dists

def animate_episode(frames, plane_size, distances, out_path, fps=5):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, plane_size - 0.5)
    ax.set_ylim(-0.5, plane_size - 0.5)
    ax.set_xticks(range(plane_size))
    ax.set_yticks(range(plane_size))
    ax.grid(True, linestyle="--", linewidth=0.5)
    (drone_pt,) = ax.plot([], [], marker="o", markersize=10, linestyle="")
    (target_pt,) = ax.plot([], [], marker="X", markersize=10, linestyle="")
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        drone_pt.set_data([], [])
        target_pt.set_data([], [])
        text.set_text("")
        return drone_pt, target_pt, text

    def update(i):
        (dx, dy), (tx, ty) = frames[i]
        drone_pt.set_data([dx], [dy])
        target_pt.set_data([tx], [ty])
        if 1 <= i <= len(distances):
            text.set_text(f"t={i}  distanza={distances[i-1]:.3f}")
        else:
            text.set_text(f"t={i}")
        return drone_pt, target_pt, text

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(frames),
        interval=int(1000/max(1,fps)), blit=False
    )

    ext = os.path.splitext(out_path)[1].lower()
    try:
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(out_path, writer=writer)
        elif ext == ".gif":
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
        else:
            raise ValueError("Usa .mp4 o .gif")
        print(f"Animazione salvata in: {out_path}")
    finally:
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints", help="Cartella con policy.pt e config.npz")
    ap.add_argument("--out", default="episode.mp4", help="Output (.mp4 o .gif)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    plane, max_steps, eps, actions = load_config(args.ckpt_dir)
    qnet = QNet(n_actions=actions.shape[0]).to(DEVICE)
    qnet.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "policy.pt"), map_location=DEVICE))
    qnet.eval()

    frames, dists = generate_episode(qnet, plane, max_steps, eps, actions, seed=args.seed)
    animate_episode(frames, plane, dists, args.out, fps=args.fps)

if __name__ == "__main__":
    main()
