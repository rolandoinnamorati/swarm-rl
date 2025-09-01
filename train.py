import os, math, random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Deque
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

# -----------------------------
# Configs
# -----------------------------
PLANE_SIZE = 101
MAX_STEPS = 300
EPS_RADIUS = max(0.5, 0.005 * PLANE_SIZE) # soglia "arrivo" in distanza euclidea
SEED = 0

# Reward shaping
K_GAIN = 1.0
STEP_COST = 0.01
REACH_BONUS = 12.0
TURN_PENALTY = 0.02
REVISIT_PENALTY = 0.01
TIMEOUT_PENALTY = 1.0

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 5000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 1.5e-3
REPLAY_SIZE = 300_000
WARMUP_STEPS = 5_000
TARGET_SYNC = 2_000 # passi di training tra sync pesi target
EPS_START = 0.30
EPS_END = 0.02
EPS_DECAY_STEPS = 250_000 # decadimento lineare dell'epsilon
GRAD_CLIP = 5.0
LOG_INTERVAL = max(1, EPISODES // 10)

# Verbosity
VERBOSE = True
LOG_STEPS = 200
EVAL_EVERY = 200
EVAL_EPISODES = 3
RET_WINDOW_EP = 50

rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

ACTIONS = np.array([
    ( 0, -1), ( 0,  1), (-1,  0), ( 1,  0),
    (-1, -1), (-1,  1), ( 1, -1), ( 1,  1),
    ( 0,  0),
], dtype=np.int32)
N_ACTIONS = len(ACTIONS)

@dataclass
class State:
    drone: Tuple[int, int]
    target: Tuple[int, int]
    t: int

class DroneEnv:
    def __init__(self, plane=PLANE_SIZE, max_steps=MAX_STEPS):
        self.plane = plane
        self.max_steps = max_steps
        self.state: State = None
        self.prev_dist: float = None
        self.last_action = None
        self.visited = set()
        self.max_dist = math.hypot(self.plane - 1, self.plane - 1)

    def _dist(self, a, b) -> float:
        return float(np.hypot(b[0]-a[0], b[1]-a[1]))

    def _norm_obs(self, s: State) -> np.ndarray:
        d = np.array([s.drone[0], s.drone[1], s.target[0], s.target[1]], dtype=np.float32)
        denom = max(1, self.plane - 1)
        return d / denom

    def reset(self) -> np.ndarray:
        d = (rng.integers(0, self.plane), rng.integers(0, self.plane))
        t = (rng.integers(0, self.plane), rng.integers(0, self.plane))
        while t == d:
            t = (rng.integers(0, self.plane), rng.integers(0, self.plane))
        self.state = State(drone=d, target=t, t=0)
        self.prev_dist = self._dist(d, t)
        self.prev_dist_norm = self.prev_dist / self.max_dist
        self.last_action = None
        self.visited = {d}
        return self._norm_obs(self.state)

    def step(self, action_id: int):
        dx, dy = ACTIONS[action_id]
        x, y = self.state.drone
        nx = int(np.clip(x + dx, 0, self.plane - 1))
        ny = int(np.clip(y + dy, 0, self.plane - 1))
        t_next = self.state.t + 1

        tx, ty = self.state.target
        d_curr = self._dist((nx, ny), (tx, ty))
        d_curr_norm = d_curr / self.max_dist
        delta = self.prev_dist_norm - d_curr_norm

        reward = K_GAIN * delta - STEP_COST

        # penalità cambio direzione
        if self.last_action is not None and tuple(ACTIONS[action_id]) != tuple(self.last_action):
            reward -= TURN_PENALTY
        self.last_action = tuple(ACTIONS[action_id])

        new_pos = (nx, ny)
        if new_pos in self.visited:
            reward -= REVISIT_PENALTY
        else:
            self.visited.add(new_pos)

        done = False
        if d_curr <= EPS_RADIUS:
            leftover = self.max_steps - t_next
            reward += REACH_BONUS * (1.0 + leftover / self.max_steps)
            done = True
        elif t_next >= self.max_steps:
            reward -= TIMEOUT_PENALTY
            done = True

        self.state = State(drone=new_pos, target=(tx, ty), t=t_next)
        self.prev_dist = d_curr
        self.prev_dist_norm = d_curr_norm
        return self._norm_obs(self.state), float(reward), bool(done), {"distance": d_curr}

    @torch.no_grad()
    def mini_eval(qnet, plane, max_steps, eps, actions, episodes=3, device=DEVICE):
        def norm_obs(state):
            (dx, dy), (tx, ty) = state
            denom = max(1, plane - 1)
            return np.array([dx, dy, tx, ty], dtype=np.float32) / denom

        rng = np.random.default_rng(123)

        def reset_state():
            d = (rng.integers(0, plane), rng.integers(0, plane))
            t = (rng.integers(0, plane), rng.integers(0, plane))
            while d == t:
                t = (rng.integers(0, plane), rng.integers(0, plane))
            return (d, t)

        successes, rets, steps_list = 0, [], []
        for _ in range(episodes):
            state = reset_state()
            prev_dist = float(np.hypot(state[1][0] - state[0][0], state[1][1] - state[0][1]))
            t = 0
            ep_ret = 0.0
            done = False
            while not done:
                obs = norm_obs(state)
                q = qnet(torch.from_numpy(obs).to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                a_id = int(np.argmax(q))
                ax, ay = actions[a_id]
                (x, y), (tx, ty) = state
                nx = int(np.clip(x + ax, 0, plane - 1))
                ny = int(np.clip(y + ay, 0, plane - 1))
                t += 1
                state = ((nx, ny), (tx, ty))
                dist = float(np.hypot(tx - nx, ty - ny))
                # reward “compatibile” con shaping normalizzato (approssimazione)
                max_dist = float(np.hypot(plane - 1, plane - 1))
                delta = (prev_dist / max_dist) - (dist / max_dist)
                r = K_GAIN * delta - STEP_COST
                ep_ret += r
                prev_dist = dist
                done = (dist <= eps) or (t >= max_steps)
            successes += int(prev_dist <= eps)
            rets.append(ep_ret)
            steps_list.append(t)
        return {
            "success_rate": successes / episodes,
            "avg_return": float(np.mean(rets)),
            "avg_steps": float(np.mean(steps_list)),
        }

class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=N_ACTIONS):
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

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.uint8))
    def __len__(self):
        return len(self.buf)

def linear_epsilon(step: int):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / float(EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac

def select_action(qnet: QNet, obs: np.ndarray, step: int):
    eps = linear_epsilon(step)
    if random.random() < eps:
        return random.randrange(N_ACTIONS), eps
    with torch.no_grad():
        q = qnet(torch.from_numpy(obs).to(DEVICE).unsqueeze(0))  # [1, nA]
        a = int(q.argmax(dim=1).item())
    return a, eps

def train():
    env = DroneEnv(PLANE_SIZE, MAX_STEPS)
    qnet = QNet().to(DEVICE)
    tgt  = QNet().to(DEVICE)
    tgt.load_state_dict(qnet.state_dict())
    optim = torch.optim.Adam(qnet.parameters(), lr=LR)
    rb = ReplayBuffer(REPLAY_SIZE)

    global_step = 0
    returns_window = deque(maxlen=LOG_INTERVAL)
    successes = 0

    t0 = time.time()
    ret_hist = deque(maxlen=RET_WINDOW_EP)

    for ep in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            a, eps = select_action(qnet, obs, global_step)
            next_obs, r, done, info = env.step(a)

            rb.push(obs, a, r, next_obs, done)
            obs = next_obs
            ep_return += r
            global_step += 1

            if VERBOSE and (global_step % LOG_STEPS == 0):
                eta_s = (time.time() - t0) / max(1, global_step) * (EPISODES * MAX_STEPS - global_step)
                print(f"[step {global_step}] ep={ep} t={env.state.t} "
                      f"dist={env.prev_dist:.2f} r_last={r:+.3f} "
                      f"eps={linear_epsilon(global_step):.3f} ETA~{int(eta_s) // 60}m")

            # update
            if len(rb) >= max(BATCH_SIZE, WARMUP_STEPS):
                s, a_b, r_b, s2, d_b = rb.sample(BATCH_SIZE)
                s_t = torch.from_numpy(s).float().to(DEVICE)
                a_t = torch.from_numpy(a_b).long().to(DEVICE)
                r_t = torch.from_numpy(r_b).float().to(DEVICE)
                s2_t = torch.from_numpy(s2).float().to(DEVICE)
                d_t = torch.from_numpy(d_b).float().to(DEVICE)

                q = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_a = qnet(s2_t).argmax(dim=1, keepdim=True)
                    q2 = tgt(s2_t).gather(1, next_a).squeeze(1)
                    target = r_t + GAMMA * (1.0 - d_t) * q2

                loss = F.smooth_l1_loss(q, target)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP)
                optim.step()

                if global_step % TARGET_SYNC == 0:
                    tgt.load_state_dict(qnet.state_dict())

        # stats
        returns_window.append(ep_return)
        if env.prev_dist <= EPS_RADIUS:
            successes += 1

        ret_hist.append(ep_return)
        if ep % LOG_INTERVAL == 0:
            avg_ret = float(np.mean(returns_window))
            rate = successes / ep
            print(f"[Ep {ep:4d}] avg_return={avg_ret:.3f}  eps={linear_epsilon(global_step):.3f}  success_rate={rate:.2%}")

        if EVAL_EVERY and (ep % EVAL_EVERY == 0):
            stats = mini_eval(qnet, PLANE_SIZE, MAX_STEPS, EPS_RADIUS, ACTIONS, episodes=EVAL_EPISODES)
            print(f"   ↳ mini-eval: success={stats['success_rate']:.2%} "
                  f"| avg_steps={stats['avg_steps']:.1f} | avg_return={stats['avg_return']:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(qnet.state_dict(), "checkpoints/policy.pt")
    np.savez_compressed(
        "checkpoints/config.npz",
        plane_size=PLANE_SIZE,
        max_steps=MAX_STEPS,
        eps=EPS_RADIUS,
        actions=ACTIONS,
        meta=np.array(["DQN v1: 2-layer MLP, Double DQN, target sync"], dtype=object),
    )
    print("Salvati: checkpoints/policy.pt e checkpoints/config.npz")

if __name__ == "__main__":
    train()