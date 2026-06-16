# Wumpus World: 50-episode recorded re-run (Snellius)

One-off sbatch that runs 50 episodes recording all of them, writing to separate
output dirs so the existing 1000-episode run in `data/results/wumpus_world/`
and `data/trajectories/wumpus_world/` is untouched.

Episodes are seeded `seed + i`, so these 50 episodes reproduce exactly the
first 50 episodes of the existing 1000-episode run (same `--seed 0`). If none
of the first 50 succeeded in the old run, none will here either — check the
old results JSON first, or change `--seed`.

**Submit from the project root on Snellius** (the `--wrap` command uses paths
relative to the submit dir), and make sure `logs/` exists:

```bash
cd /path/to/MinigridMP-AIF-JAX
mkdir -p logs
```

```bash
sbatch --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --mem=16G --time=00:30:00 \
  --job-name=ww-rec50-active-inference --output=logs/%x_%j.out \
  --wrap='source cluster/setup_env.sh && JAX_PLATFORMS=cuda python run_wumpus_world.py \
    --planning-method active-inference --grid-size 5 --n-configs 25 --n-pits 4 \
    --obs-noise 0.1 --pos-noise 0.05 --slip-prob 0.01 --episodes 50 --max-steps 16 \
    --planning-horizon 8 --planning-iterations 150 --damping 0.6 --sense-cost 0.7 \
    --pit-penalty 1.0 --wumpus-penalty 1.0 --goal-temperature 1.0 --seed 0 --receding-horizon \
    --output data/results/wumpus_world_record50/active-inference.json \
    --record-trajectories 50 --trajectory-dir data/trajectories/wumpus_world_record50/active-inference'
```

For other methods, change `--planning-method`, `--damping`, the job name, and
the output paths. Per-method damping from `params.yaml` (`planning_iterations`
is 150 for all):

| method           | damping |
|------------------|---------|
| loopy            | 1.0     |
| dyn-channel      | 0.25    |
| nuijten          | 1.0     |
| vbp-channel      | 0.6     |
| active-inference | 0.6     |

All methods in one loop:

```bash
for entry in loopy:1.0 dyn-channel:0.25 nuijten:1.0 vbp-channel:0.6 active-inference:0.6; do
  method="${entry%%:*}"; damping="${entry##*:}"
  sbatch --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --mem=16G --time=00:30:00 \
    --job-name="ww-rec50-${method}" --output=logs/%x_%j.out \
    --wrap="source cluster/setup_env.sh && JAX_PLATFORMS=cuda python run_wumpus_world.py \
      --planning-method ${method} --grid-size 5 --n-configs 25 --n-pits 4 \
      --obs-noise 0.1 --pos-noise 0.05 --slip-prob 0.01 --episodes 50 --max-steps 16 \
      --planning-horizon 8 --planning-iterations 150 --damping ${damping} --sense-cost 0.7 \
      --pit-penalty 1.0 --wumpus-penalty 1.0 --goal-temperature 1.0 --seed 0 --receding-horizon \
      --output data/results/wumpus_world_record50/${method}.json \
      --record-trajectories 50 --trajectory-dir data/trajectories/wumpus_world_record50/${method}"
done
```
