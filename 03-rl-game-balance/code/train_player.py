"""
Mini Whisper - MaskablePPO 训练 (本地 I/O + 无 EvalCallback)
"""
import os, sys, time, shutil
sys.path.insert(0, "/content/drive/MyDrive/mini_whisper")

import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from env import MiniWhisperEnv

LOCAL    = "/content/work"
TB_DIR   = f"{LOCAL}/tb_logs"
CKPT_DIR = f"{LOCAL}/checkpoints"
FINAL    = f"{LOCAL}/maskable_ppo_final.zip"
DRIVE_OUT = "/content/drive/MyDrive/mini_whisper/training_output"

for d in [TB_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_STEPS = 300_000
N_ENVS      = 4
SEED        = 42

def mask_fn(env):
    return env.unwrapped.action_masks()

def make_env(seed):
    def _init():
        env = MiniWhisperEnv(seed=seed)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init

def main():
    print(f"[init] torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    train_env = DummyVecEnv([make_env(SEED + i) for i in range(N_ENVS)])

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=TB_DIR,
        verbose=1,
        seed=SEED,
        device="cuda",
    )

    # 只留 Checkpoint，去掉 EvalCallback（避免训练中评估卡死）
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=CKPT_DIR,
        name_prefix="mppo",
    )

    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=ckpt_cb,
        tb_log_name="mppo_run",
        progress_bar=True,
    )
    dt = time.time() - t0
    print(f"[done] total time = {dt/60:.1f} min")

    model.save(FINAL)
    print(f"[saved local] {FINAL}")

    # 训练结束后单独评估（独立 env，跟 callback 无关）
    print("[eval] running 50 episodes ...")
    eval_env = DummyVecEnv([make_env(SEED + 999)])
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    print(f"[final eval] mean_reward = {mean_r:.2f} ± {std_r:.2f}  (50 eps)")

    # 同步到 Drive
    print(f"[sync] copying {LOCAL} → {DRIVE_OUT}")
    if os.path.exists(DRIVE_OUT):
        shutil.rmtree(DRIVE_OUT)
    shutil.copytree(LOCAL, DRIVE_OUT)
    print(f"[sync done] all artifacts in {DRIVE_OUT}")

if __name__ == "__main__":
    main()
