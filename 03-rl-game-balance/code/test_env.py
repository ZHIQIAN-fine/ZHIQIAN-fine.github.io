"""env.py 接口与基础逻辑验证"""
import numpy as np
from env import MiniWhisperEnv, NUM_ACTIONS, ACTION_END_TURN

passed = 0
total = 0

def check(name, cond):
    global passed, total
    total += 1
    print(f"{'✅' if cond else '❌'} Test {total}: {name}")
    if cond:
        passed += 1


# Test 1: gymnasium env_checker
try:
    from gymnasium.utils.env_checker import check_env
    env = MiniWhisperEnv(seed=0)
    check_env(env, skip_render_check=True)
    check("gymnasium env_checker 通过", True)
except Exception as ex:
    print(f"  ⚠ env_checker 错误: {ex}")
    check("gymnasium env_checker 通过", False)

# Test 2: reset 返回正确形状
env = MiniWhisperEnv(seed=42)
obs, info = env.reset()
check("reset 返回 obs 形状正确",
      isinstance(obs, np.ndarray) and obs.shape == (env.obs_dim,))

# Test 3: 动作空间维度
check(f"action_space 大小 = NUM_ACTIONS({NUM_ACTIONS})",
      env.action_space.n == NUM_ACTIONS)

# Test 4: action_mask 形状
mask = env.action_masks()
check("action_masks 形状正确", mask.shape == (NUM_ACTIONS,) and mask.dtype == bool)

# Test 5: 结束回合一定合法
check("结束回合(action=NUM_CARDS)合法", mask[ACTION_END_TURN])

# Test 6: 至少一张手牌可打(初始牌组都是攻击/防御 cost=1)
check("初始时至少一张手牌可打", mask[:ACTION_END_TURN].any())

# Test 7: step 返回 5 元组
obs, r, done, trunc, info = env.step(ACTION_END_TURN)
check("step 返回 5 元组", isinstance(obs, np.ndarray) and isinstance(r, float))

# Test 8: 跑一整局不崩溃
env2 = MiniWhisperEnv(seed=7)
obs, _ = env2.reset()
steps = 0
done = False
while not done and steps < 1000:
    mask = env2.action_masks()
    legal = np.where(mask)[0]
    a = int(np.random.choice(legal))
    obs, r, done, trunc, info = env2.step(a)
    steps += 1
check(f"随机 agent 跑完一整局({steps} 步,胜利={info.get('victory')})", done)

# Test 9: 多 seed 复现
env3a = MiniWhisperEnv(seed=123)
env3b = MiniWhisperEnv(seed=123)
o1, _ = env3a.reset()
o2, _ = env3b.reset()
check("相同 seed → 相同初始 obs", np.allclose(o1, o2))

# Test 10: 非法动作惩罚
env4 = MiniWhisperEnv(seed=0)
env4.reset()
mask = env4.action_masks()
illegal = None
for i in range(NUM_ACTIONS):
    if not mask[i]:
        illegal = i
        break
if illegal is not None:
    obs, r, done, trunc, info = env4.step(illegal)
    check("非法动作有惩罚 + invalid_action 标记",
          r < 0 and info.get("invalid_action", False))
else:
    check("非法动作有惩罚(跳过:无非法动作)", True)

# Test 11: render 不崩溃
env5 = MiniWhisperEnv(render_mode="ansi", seed=0)
env5.reset()
text = env5.render()
check("render('ansi') 返回字符串", isinstance(text, str) and len(text) > 0)

print(f"\n{'🎉' if passed == total else '⚠️'} {passed}/{total} 测试通过")
