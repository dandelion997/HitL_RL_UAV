# HitL_RL_UAV: Human-in-the-Loop Reinforcement Learning for UAV Obstacle Avoidance

**HitL_RL_UAV** is a research codebase for human-in-the-loop (HitL) / shared-control reinforcement learning applied to unmanned aerial vehicle (UAV) obstacle avoidance. It combines deep RL (DDPG, SAC) with flow-field planning (IFDS/IIFDS) and provides baseline comparisons (e.g., TS2C, HULA), multi‑obstacle testing utilities, trajectory smoothness metrics, and plotting scripts.


---

## Project Structure
```
HitL_RL_UAV/
├─ IIFDS-DDPG-random_start/            # DDPG training/eval (randomized starts)
├─ IIFDS-SAC-random_start/             # SAC training/eval (randomized starts)
├─ GIF/                                # Example visualizations/animations
├─ IIFDS.py                            # IIFDS flow-field planner core
├─ Method.py                           # Common utilities (rewards, helpers)
├─ dynamic_obstacle_environment.py     # Dynamic obstacle env definition
├─ Multi_obstacle_environment_test.py  # Multi-obstacle test entrypoint
├─ test_multi_TS2C.py                  # Baseline: TS2C
├─ test_multi_HULA.py                  # Baseline: HULA
├─ test_multi_our.py                   # Our HitL fusion method
├─ test_multi_RL.py                    # RL-only baseline (no fusion)
├─ test_multi_fixed_rate.py            # Control: fixed-rate fusion variant
├─ draw.py                             # Training curve plotting utilities
├─ config.py                           # Global hyperparameters and settings
├─ reset_history.sh                    # Optional: clean cached logs/history
└─ README.md
```
> Note: Some training/evaluation entrypoints may live inside the subfolders (e.g., `IIFDS-*`). See comments and inline docstrings within those directories for the exact command to launch.

---

## Requirements
- **Python**: 3.8–3.11 recommended
- **Core packages**:
  - `numpy`
  - `torch`
  - `matplotlib`
  - `seaborn==0.11.1` (for plots)
- **MATLAB (optional)**: for metrics helpers `calGs.m`, `calLs.m`

You can install via a `requirements.txt` of your own, or follow the manual install below.

---


## Quickstart

### Training
Training entrypoints exist inside `IIFDS-DDPG-random_start/` and `IIFDS-SAC-random_start/`. A common pattern is:
```bash
# DDPG (example — adjust to your actual entry script/args)
cd IIFDS-DDPG-random_start
python main.py --episodes 500 

# SAC (example — adjust to your actual entry script/args)
cd ../IIFDS-SAC-random_start
python main.py --episodes 500 
```
> Hyperparameters are defined in `config.py`.

### Evaluation & Multi-Obstacle Tests
```bash
# Root-level multi-obstacle environment evaluation
python Multi_obstacle_environment_test.py
```

### Baselines 
```bash
# Shared-control baselines
python test_multi_TS2C.py     # TS2C
python test_multi_HULA.py     # HULA

# Our HitL fusion
python test_multi_our.py

# RL-only baseline (no human fusion)
python test_multi_RL.py
```

### Fixed-Rate Fusion Control
```bash
python test_multi_fixed_rate.py
```

---

## Configuration
- **Global hyperparameters** are kept in `config.py` (e.g., `MAX_EPISODE`, `batch_size`, learning rates, noise parameters).  
- **Environment parameters** (e.g., obstacle count/speeds, refresh rules) are defined in `dynamic_obstacle_environment.py`.

---

## Metrics 
- **Trajectory smoothness (Gs/Ls)**: MATLAB helpers `calGs.m` and `calLs.m` compute smoothness/turn metrics on saved trajectories.
- **Additional KPIs**: reward, success rate, path length, steps-to-goal, and threat index.

