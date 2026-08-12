"""enemies.py 测试脚本"""
import random
from enemies import ENEMIES, NUM_ENEMIES, make_enemy, roll_intent, get_enemy_pool

passed = 0
total = 0

def check(name, cond):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f"✅ Test {total}: {name}")
    else:
        print(f"❌ Test {total}: {name}")


# Test 1
check("敌人数量为 6", NUM_ENEMIES == 6)

# Test 2
check("所有敌人有 name/hp/intents 字段",
      all("name" in e and "hp" in e and "intents" in e for e in ENEMIES.values()))

# Test 3
check("Boss(id=5)是白噪本身且 HP=65",
      ENEMIES[5]["name"] == "白噪本身" and ENEMIES[5]["hp"] == 65)

# Test 4
e = make_enemy(0, rng=random.Random(42))
check("make_enemy 创建实例:HP=20, armor=0",
      e["hp"] == 20 and e["armor"] == 0 and e["max_hp"] == 20)

# Test 5
intent = roll_intent(e)
check("roll_intent 返回有效意图(含 type)", "type" in intent)

# Test 6
check("roll_intent 后 next_intent 已设置", e["next_intent"] is not None)

# Test 7
boss = make_enemy(5, rng=random.Random(0))
# 强制刷大招冷却
boss["big_attack_cd"] = 4
intent = roll_intent(boss)
check("Boss 大招冷却中不会出大招", intent["type"] != "big_attack")

# Test 8
check("get_enemy_pool(0) 返回简单敌人", set(get_enemy_pool(0)) <= {0, 1})

# Test 9
check("get_enemy_pool(3) 只返回 Boss", get_enemy_pool(3) == [5])

# Test 10
check("intent_weights 长度与 intents 相同",
      all(len(e["intents"]) == len(e["intent_weights"]) for e in ENEMIES.values()))

# Test 11
weights_ok = all(abs(sum(e["intent_weights"]) - 1.0) < 0.01 for e in ENEMIES.values())
check("intent_weights 之和约为 1.0", weights_ok)

# Test 12 — 多次 roll 不应崩溃
e2 = make_enemy(3, rng=random.Random(7))
ok = True
try:
    for _ in range(50):
        roll_intent(e2)
except Exception:
    ok = False
check("50 次 roll_intent 不崩溃", ok)

print(f"\n{'🎉' if passed == total else '⚠️'} {passed}/{total} 测试通过")
