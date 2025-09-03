import os, argparse, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=9):
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
    max_steps  = int(data["max_steps"])
    eps        = float(data["eps"])
    actions    = data["actions"]  # shape [N,2]
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
        denom = max(1, self.plane - 1)              # same normalization as train.py
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

def snap_to_target_action(drone, target, actions):
    dx = target[0] - drone[0]
    dy = target[1] - drone[1]
    sx, sy = int(np.sign(dx)), int(np.sign(dy))
    for i, (ax, ay) in enumerate(actions):
        if (ax, ay) == (sx, sy):
            return i
    want = (sx, 0) if abs(dx) >= abs(dy) else (0, sy)
    for i, (ax, ay) in enumerate(actions):
        if (ax, ay) == want:
            return i
    return 0

def greedy_with_fallback(qnet, obs, actions, raw_state, plane, flat_tol=1e-6, rng=None):
    with torch.no_grad():
        q = qnet(torch.from_numpy(obs).to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()

    if np.ptp(q) > flat_tol:
        return int(np.argmax(q))

    (dx, dy), (tx, ty) = raw_state
    curr_d2 = (tx - dx) ** 2 + (ty - dy) ** 2

    best_d2 = float("inf")
    best_idxs = []
    mover_idxs = []
    nonstay_mover_idxs = []

    for i, (ax, ay) in enumerate(actions):
        nx = int(np.clip(dx + ax, 0, plane - 1))
        ny = int(np.clip(dy + ay, 0, plane - 1))
        d2 = (tx - nx) ** 2 + (ty - ny) ** 2
        moved = (nx != dx) or (ny != dy)
        is_stay = (ax == 0 and ay == 0)

        if d2 < best_d2 - 1e-12:
            best_d2 = d2
            best_idxs = [i]
        elif abs(d2 - best_d2) <= 1e-12:
            best_idxs.append(i)

        if moved:
            mover_idxs.append(i)
            if not is_stay:
                nonstay_mover_idxs.append(i)

    improving = [i for i in mover_idxs if ((tx - int(np.clip(dx + actions[i][0], 0, plane - 1))) ** 2 +
                                           (ty - int(np.clip(dy + actions[i][1], 0, plane - 1))) ** 2) < curr_d2]
    if improving:
        cand = improving
    else:
        cand = [i for i in best_idxs if i in nonstay_mover_idxs] or \
               [i for i in best_idxs if i in mover_idxs] or \
               best_idxs

    if rng is None:
        rng = np.random.default_rng()

    return int(rng.choice(cand))

def generate_episode(qnet, plane, max_steps, eps, actions, seed=None, snap=True, anti_stall=10):
    rng = np.random.default_rng(seed)
    env = ReplayEnv(plane, max_steps, eps, rng)
    obs = env.reset()
    frames = [env.raw_state()]
    dists = []
    done = False

    same_pos_count = 0
    last_pos = frames[0][0]

    while not done:
        (drone, target) = env.raw_state()

        # snap when adjacent (manhattan 1) or diagonal-adjacent
        if snap and ((abs(target[0]-drone[0]) + abs(target[1]-drone[1]) == 1) or
                     (abs(target[0]-drone[0]) == 1 and abs(target[1]-drone[1]) == 1)):
            a_id = snap_to_target_action(drone, target, actions)
        else:
            a_id = greedy_with_fallback(qnet, obs, actions, env.raw_state(), plane, rng=rng)

        a_vec = actions[a_id]
        obs, dist, done = env.step(a_vec)
        frames.append(env.raw_state())
        dists.append(dist)

        curr_pos = frames[-1][0]
        if curr_pos == last_pos:
            same_pos_count += 1
            if same_pos_count >= anti_stall:
                (dx, dy), (tx, ty) = env.raw_state()
                mover_idxs = []
                for i, (ax, ay) in enumerate(actions):
                    nx = int(np.clip(dx + ax, 0, plane - 1))
                    ny = int(np.clip(dy + ay, 0, plane - 1))
                    if (nx, ny) != (dx, dy):
                        mover_idxs.append(i)
                if mover_idxs:
                    best_i = min(mover_idxs, key=lambda i:
                    (tx - int(np.clip(dx + actions[i][0], 0, plane - 1))) ** 2 +
                    (ty - int(np.clip(dy + actions[i][1], 0, plane - 1))) ** 2)
                    obs, dist, done = env.step(actions[best_i])
                    frames[-1] = env.raw_state()
                    dists[-1] = dist
                same_pos_count = 0  # reset anti-stall
        else:
            same_pos_count = 0
            last_pos = curr_pos

    return frames, dists

def animate_episode(frames, plane_size, distances, out_path, fps=6):
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
    ap.add_argument("--seed", type=int, default=None, help="Seed; None = spawn diverso ogni run")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--no_snap", action="store_true", help="Disabilita lo snap-to-target in replay")
    ap.add_argument("--eps_override", type=float, default=None, help="Override EPS di arrivo (solo in replay)")
    args = ap.parse_args()

    plane, max_steps, eps, actions = load_config(args.ckpt_dir)
    if args.eps_override is not None:
        eps = float(args.eps_override)

    qnet = QNet(n_actions=actions.shape[0]).to(DEVICE)
    qnet.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "policy.pt"), map_location=DEVICE))
    qnet.eval()

    print(f"[PLAY] plane={plane} max_steps={max_steps} eps={eps} n_actions={actions.shape[0]}")

    frames, dists = generate_episode(
        qnet, plane, max_steps, eps, actions,
        seed=args.seed, snap=not args.no_snap
    )
    animate_episode(frames, plane, dists, args.out, fps=args.fps)

if __name__ == "__main__":
    main()
