import sys
sys.path.insert(0, "/content/drive/MyDrive/mini_whisper")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import MiniWhisperEnv
import numpy as np

model = MaskablePPO.load("/content/drive/MyDrive/mini_whisper/training_output/maskable_ppo_final.zip")

n = 200
floors_reached = []
rewards = []
boss_reached = 0
victories = 0

for i in range(n):
    env = MiniWhisperEnv(seed=10000 + i)
    env_wrapped = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    obs, _ = env_wrapped.reset()
    done = False
    total_r = 0
    last_info = {}
    while not done:
        mask = env_wrapped.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)                                        # ← 关键修复
        obs, r, term, trunc, info = env_wrapped.step(action)
        total_r += r
        last_info = info
        done = term or trunc
    floors_reached.append(env.floor)
    rewards.append(total_r)
    if env.floor >= 10:
        boss_reached += 1
    # 通关判定：优先看 info.victory,其次看 boss 是否被打死
    won = bool(last_info.get("victory", False))
    if not won:
        # 兜底:到达第 10 层且玩家还活着
        try:
            won = (env.floor >= 10 and env.player.alive and not env.enemies)
        except Exception:
            won = False
    if won:
        victories += 1

print(f"\n=== {n} 局评估 ===")
print(f"平均奖励        : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"最高奖励        : {max(rewards):.2f}")
print(f"最低奖励        : {min(rewards):.2f}")
print(f"平均到达层数    : {np.mean(floors_reached):.2f} / 10")
print(f"到达 boss(10层) : {boss_reached}/{n}  ({boss_reached/n*100:.1f}%)")
print(f"通关数          : {victories}/{n}  ({victories/n*100:.1f}%)")
print(f"\n层数分布:")
for f in range(1, 11):
    cnt = sum(1 for x in floors_reached if x == f)
    bar = "█" * (cnt * 40 // n)
    print(f"  Floor {f:2d}: {cnt:3d}  {bar}")
