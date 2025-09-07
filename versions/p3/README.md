```bash
python .\train.py --ckpt_dir checkpoints --L 100 --dt 0.1 --v_max 10 --a_max 10 --max_steps 400 --eps 1.0 --tgt_speed 0.5 --n_envs 16 --horizon 512 --total_steps 800000 --epochs 10 --minibatch 4096 --lr 3e-4 --ent_coef 0.01

python .\play.py --ckpt_dir checkpoints  --out episode.gif --seed 42
```