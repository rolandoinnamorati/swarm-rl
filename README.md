# 🚀 (Working Title) Decentralized Learning for Autonomous Kamikaze Drone Swarms with Integrated IFF and Target Engagement in Adversarial Environments

> ⚠️ **Ethics & Safety First.**  
> This repository is **strictly for simulation-based research** on decentralized multi-agent reinforcement learning.  
> No real-world deployment, weaponization, or harmful use is permitted or supported. Concepts like “kamikaze” and “target engagement” are treated as **non-destructive intercept/tagging** events **in simulation only** (e.g., virtual tag or contact).

## 📌 Overview
This project explores **decentralized multi-agent reinforcement learning (MARL)** for swarms of autonomous drones operating in **adversarial** simulated environments.  

My long-term vision is to learn **emergent coordination** (formation, target allocation, deconfliction) under **limited local observations** and **noisy/limited communications**, with an **IFF-like** (Identification-Friend-or-Foe) *simulated* module to avoid friendly interference and to prioritize valid targets.

At this very early stage (Aug, 2025), the repo contains a **single-agent baseline** on a 2D grid to validate training loops, reward shaping, and evaluation tooling. MARL, IFF simulation, and adversarial behaviors will be introduced progressively.

---

## ✨ Goals (High-Level)
- **Decentralization:** independent agents with local observations; optional, bandwidth-limited comms.
- **Emergent coordination:** task/target assignment, formation, and collision avoidance **without a central controller**.
- **Adversarial simulation:** moving/strategic targets, distractors, and counter-measures modeled in sim.
- **Robustness & generalization:** curriculum learning, noise/latency, domain randomization.
- **Safety constraints:** IFF-like *simulated* classification to avoid friendly “engagement” (non-destructive tagging).

---

## 🧪 Current Baseline (v0)
A minimal **DQN** (PyTorch) agent learns to intercept a **static target** on a discrete 2D grid.

**Key pieces**
- Reward shaping: positive when distance decreases (normalized), small penalties for zig-zag/revisit, arrival bonus.
- Double DQN, target network, replay buffer, epsilon-greedy with decay.
- Optional large grid, 8-direction actions + “stay”.

**Train & Play**
```bash
# 1) Create venv and install deps
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate
pip install torch numpy matplotlib

# 2) Train (saves checkpoints/policy.pt + checkpoints/config.npz)
python train.py

# 3) Replay with animation (MP4 requires ffmpeg; otherwise use GIF)
python play.py --ckpt_dir checkpoints --out episode.mp4 --fps 6
# or
python play.py --ckpt_dir checkpoints --out episode.gif --fps 6
# reproducible spawn
python play.py --seed 42
```

---

## 🗺️ Roadmap (Milestones)
- P0 — Baseline (done): single agent, static target, DQN on grid.
- P1 — Moving targets: stochastic/strategic motion; pursuit robustness.
- P2 — Multi-agent (decentralized): shared/independent policies, collision avoidance, task allocation.
- P3 — Obstacles & maps: local planning with partial observability (egocentric crop); limited/noisy comms.
- P4 — Continuous dynamics: (x,y,vx,vy) + acceleration actions; PPO/SAC; noise/latency/slip.
- P5 — Adversarial training & IFF sim: distractors, decoys, spoofing; non-destructive “engagement” logic; safety constraints.
- P6 — Evaluation & ablations: compare MARL methods (IPPO/MAPPO, MADDPG, QMIX/VDN), comms budgets, robustness to failures.

---

## 🤝 Contributing

PRs are welcome!