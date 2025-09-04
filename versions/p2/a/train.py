import os
import math
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================
# P2a — PPO on 2D continuous kinematics (x,y,vx,vy), actions (ax, ay)
# Target static. No noise/latency/slip in this phase.
# Scale is in *meters*, dt in seconds.
# =============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Continuous2DEnv:
    def __init__(self,
                 L: float = 100.0,
                 dt: float = 0.1,
                 v_max: float = 10.0,
                 a_max: float = 5.0,
                 max_steps: int = 400,
                 eps: float = 1.0,  #success threshold in meters
                 include_angle: bool = True,
                 include_time: bool = True,
                 seed: int = 0):
        self.L = float(L)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.include_angle = include_angle
        self.include_time = include_time
        self.rng = np.random.default_rng(seed)

        self.p = None  #(x, y)
        self.v = None  #(vx, vy)
        self.tgt = None  #(x_t, y_t)
        self.t = 0
        self.prev_d = None

        self.diag_max = math.sqrt(2.0) * self.L

    def _spawn(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        tgt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        while np.allclose(p, tgt, atol=1e-3):
            tgt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        return p, tgt

    def _obs(self) -> np.ndarray:
        #relative position normalized to [-1,1] roughly via division by L
        dp = (self.tgt - self.p) / max(1e-6, self.L)
        v_norm = self.v / max(1e-6, self.v_max)

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

    def reset(self) -> np.ndarray:
        self.p, self.tgt = self._spawn()
        self.v = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.prev_d = float(np.linalg.norm(self.tgt - self.p))
        return self._obs()

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -self.a_max, self.a_max)

        self.v = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
        self.p = np.clip(self.p + self.v * self.dt, 0.0, self.L)
        self.t += 1

        #distances and reward shaping
        d = float(np.linalg.norm(self.tgt - self.p))
        delta = (self.prev_d - d) / max(1e-6, self.diag_max)  # normalized progress

        alpha = 2.0
        step_cost = 1e-3
        lambda_a = 1e-3

        r = alpha * delta - step_cost - lambda_a * float(np.dot(a, a))

        done = False
        info = {"dist": d}
        #success termination
        if d <= self.eps:
            leftover = (self.max_steps - self.t) / max(1, self.max_steps)
            R = 20.0
            r += R * (1.0 + leftover)
            done = True
            info["success"] = True
        elif self.t >= self.max_steps:
            r -= 1.0  #timeout penalty
            done = True
            info["success"] = False

        self.prev_d = d
        return self._obs(), r, done, info

class VecEnv:
    def __init__(self, n_envs: int, env_ctor, env_kwargs: dict):
        self.envs = [env_ctor(**env_kwargs) for _ in range(n_envs)]
        self.n = n_envs

    def reset(self) -> np.ndarray:
        obs = [e.reset() for e in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions: np.ndarray):
        obs_next, rews, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, info = e.step(a)
            if d:
                o = e.reset()  # auto-reset for rollout collection
            obs_next.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return np.stack(obs_next, axis=0), np.asarray(rews, dtype=np.float32), np.asarray(dones, dtype=np.bool_), infos

class RunningNorm:
    def __init__(self, shape, eps=1e-5, clip=5.0):
        self.mean = torch.zeros(shape, dtype=TORCH_DTYPE, device=DEVICE)
        self.var = torch.ones(shape, dtype=TORCH_DTYPE, device=DEVICE)
        self.count = eps
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count

        self.mean = new_mean
        self.var = new_var.clamp_min(1e-8)
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / torch.sqrt(self.var)
        #return x.clamp_(-self.clip, self.clip)
        return torch.clamp(x, -self.clip, self.clip)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        log_std = self.log_std.clamp(-3.0, 2.0)
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

class TanhNormal:
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor):
        self.mu = mu
        self.log_std = log_std
        self.std = torch.exp(log_std)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(self.mu)
        z = self.mu + self.std * eps
        a = torch.tanh(z)
        log_prob = self._log_prob_from_z(z, a)
        return a, log_prob, z

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.mu)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        # a in (-1,1)
        # a = a.clamp(-0.999999, 0.999999)
        # z = torch.atanh(a)
        # return self._log_prob_from_z(z, a)
        a_clamped = torch.clamp(a, -0.999999, 0.999999)
        z = torch.atanh(a_clamped)
        return self._log_prob_from_z(z, a_clamped)

    def _log_prob_from_z(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        log_unnormalized = -0.5 * ((z - self.mu) / (self.std + 1e-8))**2
        log_norm = self.log_std + 0.5 * math.log(2 * math.pi)
        log_normal = (log_unnormalized - log_norm).sum(-1)
        log_correction = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return log_normal - log_correction

@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 16
    horizon: int = 512 #steps per env per update → batch = n_envs*horizon
    epochs: int = 10
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.7
    eval_every: int = 50_000
    save_every: int = 200_000
    seed: int = 0

@dataclass
class EnvConfig:
    L: float = 100.0
    dt: float = 0.1
    v_max: float = 10.0
    a_max: float = 5.0
    max_steps: int = 400
    eps: float = 1.0
    include_angle: bool = True
    include_time: bool = True

class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = torch.zeros((size, obs_dim), dtype=TORCH_DTYPE, device=DEVICE)
        self.actions = torch.zeros((size, act_dim), dtype=TORCH_DTYPE, device=DEVICE)
        self.logprobs = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.values = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.rewards = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.dones = torch.zeros(size, dtype=torch.bool, device=DEVICE)
        self.ptr = 0
        self.max = size
        self.values_next = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)

    def add(self, o, a, lp, v, v_next, r, d):
        i = self.ptr
        self.obs[i] = o
        self.actions[i] = a
        self.logprobs[i] = lp
        self.values[i] = v
        self.values_next[i] = v_next
        self.rewards[i] = r
        self.dones[i] = d
        self.ptr += 1

    def full(self):
        return self.ptr >= self.max

@torch.no_grad()
def compute_gae(buf: RolloutBuffer, gamma: float, lam: float):
    size = buf.ptr
    adv = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)

    next_adv = torch.zeros(1, dtype=TORCH_DTYPE, device=DEVICE)

    for t in reversed(range(size)):
        mask = 1.0 - buf.dones[t].float()
        delta = buf.rewards[t] + gamma * buf.values_next[t] * mask - buf.values[t]
        next_adv = delta + gamma * lam * mask * next_adv
        adv[t] = next_adv

    ret = adv + buf.values[:size]
    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

#Evaluation rollouts (deterministic policy)
@torch.no_grad()
def evaluate(actor: Actor, critic: Critic, env_cfg: EnvConfig, obs_norm: RunningNorm, episodes: int = 10):
    succ = 0
    steps_list = []
    for _ in range(episodes):
        env = Continuous2DEnv(**env_cfg.__dict__)
        o = env.reset()
        done = False
        t = 0
        while not done:
            obs_t = torch.tensor(o, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
            obs_t = obs_norm.normalize(obs_t)
            mu, log_std = actor(obs_t)
            dist = TanhNormal(mu, log_std)
            a = dist.deterministic()
            # scale to [-a_max, a_max]
            a_scaled = a * env_cfg.a_max
            o, r, done, info = env.step(a_scaled.squeeze(0).cpu().numpy())
            t += 1
        succ += int(info.get("success", False))
        steps_list.append(t)
    return {
        "success_rate": succ / episodes,
        "avg_steps": float(np.mean(steps_list)),
    }

def main():
    parser = argparse.ArgumentParser()
    #Env
    parser.add_argument("--L", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--v_max", type=float, default=10.0)
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--no_angle", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    #PPO
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.7)
    parser.add_argument("--eval_every", type=int, default=50_000)
    parser.add_argument("--save_every", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_cont")

    args = parser.parse_args()

    set_global_seeds(args.seed)

    env_cfg = EnvConfig(L=args.L, dt=args.dt, v_max=args.v_max, a_max=args.a_max,
                        max_steps=args.max_steps, eps=args.eps,
                        include_angle=(not args.no_angle), include_time=(not args.no_time))

    ppo_cfg = PPOConfig(total_steps=args.total_steps, n_envs=args.n_envs, horizon=args.horizon,
                        epochs=args.epochs, minibatch_size=args.minibatch, gamma=args.gamma,
                        gae_lambda=args.gae_lambda, clip_ratio=args.clip, lr=args.lr,
                        ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                        eval_every=args.eval_every, save_every=args.save_every, seed=args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    vec = VecEnv(ppo_cfg.n_envs, Continuous2DEnv, env_cfg.__dict__)
    obs = vec.reset()

    tmp_env = Continuous2DEnv(**env_cfg.__dict__)
    obs_dim = tmp_env.reset().shape[0]
    act_dim = 2

    actor = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)

    optim_actor = optim.Adam(actor.parameters(), lr=ppo_cfg.lr)
    optim_critic = optim.Adam(critic.parameters(), lr=ppo_cfg.lr)

    obs_norm = RunningNorm((obs_dim,))

    batch_size = ppo_cfg.n_envs * ppo_cfg.horizon
    buf = RolloutBuffer(obs_dim, act_dim, batch_size)

    global_steps = 0
    start_time = time.time()

    #For evaluation scheduling
    next_eval = ppo_cfg.eval_every
    next_save = ppo_cfg.save_every

    while global_steps < ppo_cfg.total_steps:
        buf.ptr = 0
        ep_rews = []

        for t in range(ppo_cfg.horizon):
            obs_t = torch.tensor(obs, dtype=TORCH_DTYPE, device=DEVICE)
            obs_norm.update(obs_t)
            obs_t = obs_norm.normalize(obs_t)

            mu, log_std = actor(obs_t)
            dist = TanhNormal(mu, log_std)
            a_t, logp_t, z_t = dist.sample()
            a_scaled = a_t * env_cfg.a_max  #tcale to action bounds

            v_t = critic(obs_t)

            actions_np = a_scaled.detach().cpu().numpy()
            obs_next, rew, done, infos = vec.step(actions_np)

            obs_next_t = torch.tensor(obs_next, dtype=TORCH_DTYPE, device=DEVICE)
            obs_next_t_norm = obs_norm.normalize(obs_next_t)
            v_next = critic(obs_next_t_norm)

            for i in range(ppo_cfg.n_envs):
                #buf.add(obs_t[i], a_t[i], logp_t[i], v_t[i], torch.tensor(rew[i], device=DEVICE), torch.tensor(done[i], device=DEVICE))
                buf.add(
                    obs_t[i].detach(),
                    a_t[i].detach(),
                    logp_t[i].detach(),
                    v_t[i].detach(),
                    v_next[i].detach(),
                    torch.tensor(rew[i], dtype=torch.float32, device=DEVICE),
                    torch.tensor(done[i], dtype=torch.bool, device=DEVICE),
                )

            obs = obs_next
            global_steps += ppo_cfg.n_envs

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=TORCH_DTYPE, device=DEVICE)
            obs_t = obs_norm.normalize(obs_t)
            last_values = critic(obs_t)
        adv, ret = compute_gae(buf, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        #Flatten batch
        O = buf.obs[:buf.ptr]
        A = buf.actions[:buf.ptr]
        LOGP = buf.logprobs[:buf.ptr]
        V = buf.values[:buf.ptr]
        ADV = adv
        RET = ret

        # PPO updates (minibatch SGD)
        idx = torch.randperm(buf.ptr, device=DEVICE)
        for epoch in range(ppo_cfg.epochs):
            for start in range(0, buf.ptr, ppo_cfg.minibatch_size):
                end = start + ppo_cfg.minibatch_size
                mb_idx = idx[start:end]
                o_mb = O[mb_idx]
                a_mb = A[mb_idx]
                old_logp_mb = LOGP[mb_idx]
                adv_mb = ADV[mb_idx]
                ret_mb = RET[mb_idx]

                mu, log_std = actor(o_mb)
                dist = TanhNormal(mu, log_std)
                logp = dist.log_prob(a_mb)
                entropy = (-logp).mean()
                ratio = torch.exp(logp - old_logp_mb)
                clipped = torch.clamp(ratio, 1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio) * adv_mb
                loss_pi = -(torch.min(ratio * adv_mb, clipped)).mean() - ppo_cfg.ent_coef * entropy

                v_pred = critic(o_mb)
                loss_v = 0.5 * ((v_pred - ret_mb) ** 2).mean() * ppo_cfg.vf_coef

                optim_actor.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), ppo_cfg.max_grad_norm)
                optim_actor.step()

                optim_critic.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), ppo_cfg.max_grad_norm)
                optim_critic.step()

        # Logging / Eval / Save
        if global_steps >= next_eval:
            stats = evaluate(actor, critic, env_cfg, obs_norm, episodes=10)
            elapsed = time.time() - start_time
            print(f"[steps {global_steps}] eval: success={stats['success_rate']*100:.1f}% "
                  f"avg_steps={stats['avg_steps']:.1f}  "
                  f"elapsed={elapsed/60:.1f}m")
            next_eval += ppo_cfg.eval_every

        if global_steps >= next_save:
            # save checkpoints + config
            torch.save(actor.state_dict(), os.path.join(args.ckpt_dir, "ppo_actor.pt"))
            torch.save(critic.state_dict(), os.path.join(args.ckpt_dir, "ppo_critic.pt"))
            np.savez(os.path.join(args.ckpt_dir, "config_continuous.npz"),
                     L=env_cfg.L, dt=env_cfg.dt, v_max=env_cfg.v_max, a_max=env_cfg.a_max,
                     max_steps=env_cfg.max_steps, eps=env_cfg.eps,
                     include_angle=env_cfg.include_angle, include_time=env_cfg.include_time,
                     obs_dim=obs_dim, act_dim=act_dim)
            print(f"[steps {global_steps}] saved checkpoints to {args.ckpt_dir}")
            next_save += ppo_cfg.save_every

    # Final save
    torch.save(actor.state_dict(), os.path.join(args.ckpt_dir, "ppo_actor.pt"))
    torch.save(critic.state_dict(), os.path.join(args.ckpt_dir, "ppo_critic.pt"))
    np.savez(os.path.join(args.ckpt_dir, "config_continuous.npz"),
             L=env_cfg.L, dt=env_cfg.dt, v_max=env_cfg.v_max, a_max=env_cfg.a_max,
             max_steps=env_cfg.max_steps, eps=env_cfg.eps,
             include_angle=env_cfg.include_angle, include_time=env_cfg.include_time,
             obs_dim=obs_dim, act_dim=act_dim)
    print(f"Training complete. Checkpoints saved to {args.ckpt_dir}")

if __name__ == "__main__":
    main()
