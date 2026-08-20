"""CPU 版训练脚本：python3 train_cpu.py <code_dir> <out_model> [steps]"""
import os, sys, time
code_dir = os.path.abspath(sys.argv[1])
out_model = os.path.abspath(sys.argv[2])
TOTAL_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
sys.path.insert(0, code_dir)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import MiniWhisperEnv

N_ENVS, SEED = 4, 42

def mask_fn(env):
    return env.unwrapped.action_masks()

def make_env(seed):
    def _init():
        return Monitor(ActionMasker(MiniWhisperEnv(seed=seed), mask_fn))
    return _init

train_env = DummyVecEnv([make_env(SEED + i) for i in range(N_ENVS)])
model = MaskablePPO(
    "MlpPolicy", train_env,
    learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=1, seed=SEED, device="cpu",
)
t0 = time.time()
model.learn(total_timesteps=TOTAL_STEPS)
print(f"[done] {(time.time()-t0)/60:.1f} min")
model.save(out_model)
print("[saved]", out_model)
