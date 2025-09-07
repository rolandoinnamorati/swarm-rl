import os
import math
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Continuous2DEnvMoving:
    def __init__(self,
                 L: float = 100.0,
                 dt: float = 0.1,
                 v_max: float = 10.0,
                 a_max: float = 5.0,
                 max_steps: int = 400,
                 eps: float = 1.0,
                 include_angle: bool = True,
                 include_time: bool = True,
                 include_tgt_vel: bool = True,
                 tgt_speed: float = 0.5,  # m/s (slow)
                 seed: int = 0):
        self.L = float(L)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.include_angle = include_angle
        self.include_time = include_time
        self.include_tgt_vel = include_tgt_vel
        self.tgt_speed = float(tgt_speed)
        self.rng = np.random.default_rng(seed)

        self.p = None
        self.v = None
        self.pt = None
        self.vt = None
        self.t = 0
        self.prev_d = None
        self.diag_max = math.sqrt(2.0) * self.L

    def _spawn_positions(self):
        p = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        pt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        # avoid identical spawn
        while np.linalg.norm(pt - p) < 1e-3:
            pt = self.rng.uniform(0.0, self.L, size=2).astype(np.float32)
        return p, pt

    def _spawn_target_velocity(self):
        theta = self.rng.uniform(0.0, 2.0 * math.pi)
        vx = math.cos(theta) * self.tgt_speed
        vy = math.sin(theta) * self.tgt_speed
        return np.array([vx, vy], dtype=np.float32)

    def _bounce(self, pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = pos
        vx, vy = vel
        if x < 0.0:
            x = 0.0
            vx = abs(vx)
        elif x > self.L:
            x = self.L
            vx = -abs(vx)
        if y < 0.0:
            y = 0.0
            vy = abs(vy)
        elif y > self.L:
            y = self.L
            vy = -abs(vy)
        return np.array([x, y], dtype=np.float32), np.array([vx, vy], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        dp = (self.pt - self.p) / max(1e-6, self.L)
        v_norm = self.v / max(1e-6, self.v_max)
        feats = [dp[0], dp[1], v_norm[0], v_norm[1]]
        if self.include_tgt_vel:
            vt_norm = self.vt / max(1e-6, self.v_max)
            feats += [vt_norm[0], vt_norm[1]]
        if self.include_angle:
            v = self.v
            dp_raw = self.pt - self.p
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
        self.p, self.pt = self._spawn_positions()
        self.v = np.zeros(2, dtype=np.float32)
        self.vt = self._spawn_target_velocity()
        self.t = 0
        self.prev_d = float(np.linalg.norm(self.pt - self.p))
        return self._obs()

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -self.a_max, self.a_max)
        # drone integration
        self.v = np.clip(self.v + a * self.dt, -self.v_max, self.v_max)
        self.p = np.clip(self.p + self.v * self.dt, 0.0, self.L)
        #target integration (bounce)
        pt_next = self.pt + self.vt * self.dt
        pt_next, vt_next = self._bounce(pt_next, self.vt)
        self.pt, self.vt = pt_next, vt_next
        self.t += 1

        d = float(np.linalg.norm(self.pt - self.p))
        delta = (self.prev_d - d) / max(1e-6, self.diag_max)
        alpha = 2.5
        step_cost = 1e-3
        lambda_a = 1e-3
        r = alpha * delta - step_cost - lambda_a * float(np.dot(a, a))

        done = False
        info: Dict = {"dist": d}
        if d <= self.eps:
            leftover = (self.max_steps - self.t) / max(1, self.max_steps)
            R = 20.0
            r += R * (1.0 + leftover)
            done = True
            info["success"] = True
        elif self.t >= self.max_steps:
            r -= 1.0
            done = True
            info["success"] = False
        self.prev_d = d
        return self._obs(), r, done, info

class VecEnv:
    def __init__(self, n_envs: int, env_ctor, env_kwargs: dict):
        self.envs = [env_ctor(**env_kwargs) for _ in range(n_envs)]
        self.n = n_envs
    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs], axis=0)
    def step(self, actions: np.ndarray):
        obs_next, rews, dones, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, info = e.step(a)
            if d:
                o = e.reset()
            obs_next.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        return (np.stack(obs_next, axis=0),
                np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                infos)

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
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + (delta**2) * self.count * batch_count / tot) / tot
        self.mean = new_mean
        self.var = new_var.clamp_min(1e-8)
        self.count = tot
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.sqrt(self.var)
        return torch.clamp(z, -self.clip, self.clip)

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
    def sample(self):
        eps = torch.randn_like(self.mu)
        z = self.mu + self.std * eps
        a = torch.tanh(z)
        log_prob = self._log_prob_from_z(z, a)
        return a, log_prob
    def deterministic(self):
        return torch.tanh(self.mu)
    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        a_clamped = torch.clamp(a, -0.999999, 0.999999)
        z = torch.atanh(a_clamped)
        return self._log_prob_from_z(z, a_clamped)
    def _log_prob_from_z(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        log_unnorm = -0.5 * ((z - self.mu) / (self.std + 1e-8))**2
        log_norm = self.log_std + 0.5 * math.log(2 * math.pi)
        log_normal = (log_unnorm - log_norm).sum(-1)
        corr = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return log_normal - corr

@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 16
    horizon: int = 512
    epochs: int = 10
    minibatch: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.005
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
    include_tgt_vel: bool = True
    tgt_speed: float = 0.5

class RolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = torch.zeros((size, obs_dim), dtype=TORCH_DTYPE, device=DEVICE)
        self.actions = torch.zeros((size, act_dim), dtype=TORCH_DTYPE, device=DEVICE)
        self.logprobs = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.values = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.values_next = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.rewards = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)
        self.dones = torch.zeros(size, dtype=torch.bool, device=DEVICE)
        self.ptr = 0
        self.max = size
    def add(self, o, a, lp, v, v_next, r, d):
        i = self.ptr
        self.obs[i] = o.detach()
        self.actions[i] = a.detach()
        self.logprobs[i] = lp.detach()
        self.values[i] = v.detach()
        self.values_next[i] = v_next.detach()
        self.rewards[i] = r
        self.dones[i] = d
        self.ptr += 1
    def full(self):
        return self.ptr >= self.max

@torch.no_grad()
def compute_gae_interleaved(buf, n_envs: int, horizon: int, gamma: float, lam: float):
    size = buf.ptr
    assert size == n_envs * horizon, "buffer pieno atteso: horizon*n_envs"
    adv = torch.zeros(size, dtype=TORCH_DTYPE, device=DEVICE)

    for e in range(n_envs):
        next_adv = torch.zeros(1, dtype=TORCH_DTYPE, device=DEVICE)
        for t in reversed(range(horizon)):
            idx = t * n_envs + e
            mask  = 1.0 - buf.dones[idx].float()
            delta = buf.rewards[idx] + gamma * buf.values_next[idx] * mask - buf.values[idx]
            next_adv = delta + gamma * lam * mask * next_adv
            adv[idx] = next_adv

    ret = adv + buf.values[:size]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

@torch.no_grad()
def evaluate(actor: Actor, env_cfg: EnvConfig, obs_norm: RunningNorm, episodes: int = 10):
    succ = 0
    steps_list = []
    a_norm_sum = 0.0
    v_norm_sum = 0.0
    vt_norm_sum = 0.0
    step_count = 0
    improve_count = 0

    for _ in range(episodes):
        env = Continuous2DEnvMoving(**env_cfg.__dict__)
        o = env.reset()
        done = False
        prev_d = None
        t = 0
        while not done:
            obs_t = torch.tensor(o, dtype=TORCH_DTYPE, device=DEVICE).unsqueeze(0)
            obs_t = obs_norm.normalize(obs_t)
            mu, log_std = actor(obs_t)
            dist = TanhNormal(mu, log_std)
            a = (dist.deterministic() * env_cfg.a_max).squeeze(0).cpu().numpy()
            o, _, done, info = env.step(a)

            a_norm_sum += float(np.linalg.norm(a))
            v_norm_sum += float(np.linalg.norm(env.v))
            vt_norm_sum += float(np.linalg.norm(env.vt))

            d_cur = info.get("dist", None)
            if d_cur is not None and prev_d is not None:
                improve_count += int(d_cur < prev_d - 1e-9)
                step_count += 1
            prev_d = d_cur

            t += 1
        succ += int(info.get("success", False))
        steps_list.append(t)

    total_steps = max(1, sum(steps_list))
    return {
        "success_rate": succ / episodes,
        "avg_steps": float(np.mean(steps_list)),
        "median_steps": float(np.median(steps_list)),
        "mean_a_norm": a_norm_sum / total_steps,
        "mean_v_norm": v_norm_sum / total_steps,
        "mean_vt_norm": vt_norm_sum / total_steps,
        "improve_rate": (improve_count / max(1, step_count)),
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
    parser.add_argument("--tgt_speed", type=float, default=0.5)
    parser.add_argument("--no_angle", action="store_true")
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--no_tgt_vel", action="store_true")

    # PPO
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
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_p3")

    args = parser.parse_args()

    set_global_seeds(args.seed)

    env_cfg = EnvConfig(L=args.L, dt=args.dt, v_max=args.v_max, a_max=args.a_max,
                        max_steps=args.max_steps, eps=args.eps,
                        include_angle=(not args.no_angle), include_time=(not args.no_time),
                        include_tgt_vel=(not args.no_tgt_vel), tgt_speed=args.tgt_speed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Build vectorized env
    vec = VecEnv(args.n_envs, Continuous2DEnvMoving, env_cfg.__dict__)
    obs = vec.reset()

    # Obs/Act dims
    tmp_env = Continuous2DEnvMoving(**env_cfg.__dict__)
    obs_dim = tmp_env.reset().shape[0]
    act_dim = 2

    actor = Actor(obs_dim, act_dim).to(DEVICE)
    critic = Critic(obs_dim).to(DEVICE)
    optim_actor = optim.Adam(actor.parameters(), lr=args.lr)
    optim_critic = optim.Adam(critic.parameters(), lr=args.lr)

    obs_norm = RunningNorm((obs_dim,))
    batch_size = args.n_envs * args.horizon
    buf = RolloutBuffer(obs_dim, act_dim, batch_size)

    global_steps = 0
    start_time = time.time()
    next_eval = args.eval_every
    next_save = args.save_every

    while global_steps < args.total_steps:
        buf.ptr = 0
        prev_dists = np.full(args.n_envs, np.nan, dtype=np.float64)
        roll_improve = 0
        roll_steps = 0
        roll_a_norm_sum = 0.0
        roll_v_norm_sum = 0.0
        roll_vt_norm_sum = 0.0
        roll_count = 0
        term_success = 0
        term_timeout = 0

        for t in range(args.horizon):
            obs_t = torch.tensor(obs, dtype=TORCH_DTYPE, device=DEVICE)
            obs_norm.update(obs_t)
            obs_tn = obs_norm.normalize(obs_t)

            mu, log_std = actor(obs_tn)
            dist = TanhNormal(mu, log_std)
            a_t, logp_t = dist.sample()
            a_scaled = a_t * args.a_max
            v_t = critic(obs_tn)

            actions_np = a_scaled.detach().cpu().numpy()
            obs_next, rew, done, infos = vec.step(actions_np)

            obs_next_t = torch.tensor(obs_next, dtype=TORCH_DTYPE, device=DEVICE)
            obs_next_tn = obs_norm.normalize(obs_next_t)
            v_next = critic(obs_next_tn)

            for i in range(args.n_envs):
                buf.add(obs_tn[i], a_t[i], logp_t[i], v_t[i], v_next[i],
                        torch.tensor(rew[i], dtype=torch.float32, device=DEVICE),
                        torch.tensor(done[i], dtype=torch.bool, device=DEVICE))

            a_norms = np.linalg.norm(actions_np, axis=1)
            roll_a_norm_sum += float(a_norms.sum())
            # try to infer velocities from obs_next (indices: dp[0:2], v[2:4], vt[4:6] if included)
            try:
                start_v = 2
                v_arr = obs_next[:, start_v:start_v+2] * args.v_max
                vt_arr = obs_next[:, start_v+2:start_v+4] * args.v_max if env_cfg.include_tgt_vel else np.zeros_like(v_arr)
                roll_v_norm_sum += float(np.linalg.norm(v_arr, axis=1).sum())
                roll_vt_norm_sum += float(np.linalg.norm(vt_arr, axis=1).sum())
            except Exception:
                pass
            roll_count += args.n_envs

            for i in range(args.n_envs):
                d_cur = float(infos[i].get("dist", np.nan)) if isinstance(infos[i], dict) else np.nan
                if not np.isnan(prev_dists[i]):
                    if not np.isnan(d_cur) and d_cur < prev_dists[i] - 1e-9:
                        roll_improve += 1
                    roll_steps += 1
                prev_dists[i] = d_cur
                if bool(done[i]):
                    if isinstance(infos[i], dict) and infos[i].get("success", False):
                        term_success += 1
                    else:
                        term_timeout += 1

            obs = obs_next
            global_steps += args.n_envs

        mean_a = roll_a_norm_sum / max(1, roll_count)
        mean_v = roll_v_norm_sum / max(1, roll_count)
        mean_vt = roll_vt_norm_sum / max(1, roll_count)
        improve_rate = roll_improve / max(1, roll_steps)
        print(f"[rollout] improve_rate={improve_rate:.1%} mean|a|={mean_a:.3f} mean|v|={mean_v:.3f} mean|vt|={mean_vt:.3f} term_success={term_success} term_timeout={term_timeout}")

        # GAE and PPO updates
        adv, ret = compute_gae_interleaved(buf, args.n_envs, args.horizon, args.gamma, args.gae_lambda)
        O = buf.obs[:buf.ptr]
        A = buf.actions[:buf.ptr]
        LOGP = buf.logprobs[:buf.ptr]
        ADV = adv
        RET = ret

        idx = torch.randperm(buf.ptr, device=DEVICE)
        for _ in range(args.epochs):
            for start in range(0, buf.ptr, args.minibatch):
                end = start + args.minibatch
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
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_mb
                loss_pi = -(torch.min(ratio * adv_mb, clipped)).mean() - args.ent_coef * entropy

                v_pred = critic(o_mb)
                loss_v = 0.5 * ((v_pred - ret_mb) ** 2).mean() * args.vf_coef

                optim_actor.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optim_actor.step()

                optim_critic.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                optim_critic.step()

        if global_steps >= next_eval:
            stats = evaluate(actor, env_cfg, obs_norm, episodes=10)
            elapsed = time.time() - start_time
            print(f"[steps {global_steps}] eval: success={stats['success_rate']*100:.1f}% avg_steps={stats['avg_steps']:.1f} median_steps={stats['median_steps']:.1f} "
                  f"mean|a|={stats['mean_a_norm']:.3f} mean|v|={stats['mean_v_norm']:.3f} mean|vt|={stats['mean_vt_norm']:.3f} improve_rate={stats['improve_rate']*100:.1f}% elapsed={elapsed/60:.1f}m")
            next_eval += args.eval_every

        if global_steps >= next_save:
            torch.save(actor.state_dict(), os.path.join(args.ckpt_dir, "ppo_actor.pt"))
            torch.save(critic.state_dict(), os.path.join(args.ckpt_dir, "ppo_critic.pt"))
            np.savez(os.path.join(args.ckpt_dir, "config_p3.npz"),
                     L=env_cfg.L, dt=env_cfg.dt, v_max=env_cfg.v_max, a_max=env_cfg.a_max,
                     max_steps=env_cfg.max_steps, eps=env_cfg.eps,
                     include_angle=env_cfg.include_angle, include_time=env_cfg.include_time,
                     include_tgt_vel=env_cfg.include_tgt_vel, tgt_speed=env_cfg.tgt_speed,
                     obs_dim=obs_dim, act_dim=act_dim)
            # save obs stats for parity
            np.savez(os.path.join(args.ckpt_dir, "obs_stats.npz"),
                     mean=obs_norm.mean.detach().cpu().numpy(),
                     var=obs_norm.var.detach().cpu().numpy(),
                     count=np.array([obs_norm.count], dtype=np.float64))
            print(f"[steps {global_steps}] saved checkpoints and obs_stats to {args.ckpt_dir}")
            next_save += args.save_every

    torch.save(actor.state_dict(), os.path.join(args.ckpt_dir, "ppo_actor.pt"))
    torch.save(critic.state_dict(), os.path.join(args.ckpt_dir, "ppo_critic.pt"))
    np.savez(os.path.join(args.ckpt_dir, "config_p3.npz"),
             L=env_cfg.L, dt=env_cfg.dt, v_max=env_cfg.v_max, a_max=env_cfg.a_max,
             max_steps=env_cfg.max_steps, eps=env_cfg.eps,
             include_angle=env_cfg.include_angle, include_time=env_cfg.include_time,
             include_tgt_vel=env_cfg.include_tgt_vel, tgt_speed=env_cfg.tgt_speed,
             obs_dim=obs_dim, act_dim=act_dim)
    np.savez(os.path.join(args.ckpt_dir, "obs_stats.npz"),
             mean=obs_norm.mean.detach().cpu().numpy(),
             var=obs_norm.var.detach().cpu().numpy(),
             count=np.array([obs_norm.count], dtype=np.float64))
    print(f"Training complete. Checkpoints saved to {args.ckpt_dir}")

if __name__ == "__main__":
    main()