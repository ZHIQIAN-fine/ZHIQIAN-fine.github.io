"""cards.py v2 测试 - 适配 effect(ctx)->dict 协议"""
import random
from cards import (
    CARDS, NUM_CARDS, get_starter_cards, get_loot_pool, get_madness_cards,
)

passed = 0
total = 0

def check(name, cond):
    global passed, total
    total += 1
    print(f"{'✅' if cond else '❌'} Test {total}: {name}")
    if cond: passed += 1


def make_ctx(p_hp=50, p_san=30, e_hp=30, e_armor=0, hand=None, deck=None):
    return {
        "player": {
            "hp": p_hp, "max_hp": 50, "san": p_san, "max_san": 30,
            "energy": 3, "armor": 0, "strength": 0, "weak": 0,
            "corrosion": 0, "overclock_buff": 0, "foresight_turns": 0,
            "rule_penalty_multiplier": 1,
            "buffs": {}, "debuffs": {},
            "hand": hand or [],
            "deck": deck or [],
            "discard": [], "exhaust": [],
        },
        "enemy": {"hp": e_hp, "max_hp": e_hp, "armor": e_armor, "strength": 0, "weak": 0},
        "rng": random.Random(42),
        "deck": deck or [], "discard": [], "exhaust": [], "hand": hand or [],
    }


# Test 1
check("卡牌数量为 15", NUM_CARDS == 15)

# Test 2
check("所有卡有 name/cost/type/effect/dest 字段",
      all(all(k in c for k in ("name", "cost", "type", "effect", "dest")) for c in CARDS.values()))

# Test 3
starter = get_starter_cards()
check("初始牌组 10 张(5 攻 5 防)",
      len(starter) == 10 and starter.count(0) == 5 and starter.count(3) == 5)

# Test 4 — 生锈匕首
ctx = make_ctx(); r = CARDS[0]["effect"](ctx)
check(f"生锈匕首 -> 6 伤(实得 {r['damage_dealt']})", r["damage_dealt"] == 6 and ctx["enemy"]["hp"] == 24)

# Test 5 — 离子护盾
ctx = make_ctx(); r = CARDS[3]["effect"](ctx)
check(f"离子护盾 -> +5 armor(实得 {r['armor_gained']})", r["armor_gained"] == 5 and ctx["player"]["armor"] == 5)

# Test 6 — 敌人有护甲时,伤害先扣护甲
ctx = make_ctx(e_armor=4); r = CARDS[0]["effect"](ctx)
check("生锈匕首 vs 4 armor:6 伤穿 4 armor + 2 HP",
      ctx["enemy"]["armor"] == 0 and ctx["enemy"]["hp"] == 28 and r["damage_dealt"] == 6)

# Test 7 — OVERCLOCK 加成
ctx = make_ctx(); CARDS[8]["effect"](ctx)  # OVERCLOCK
check("OVERCLOCK 后 buff=3", ctx["player"]["overclock_buff"] == 3)
r = CARDS[0]["effect"](ctx)
check(f"OVERCLOCK 后 生锈匕首 = 6+3 = 9 伤", r["damage_dealt"] == 9)

# Test 8 — 冥想程序
ctx = make_ctx(p_hp=30, p_san=20); r = CARDS[4]["effect"](ctx)
check("冥想程序 -> 回血 6 + 回 SAN 1",
      r["heal"] == 6 and r["san_gain"] == 1 and ctx["player"]["hp"] == 36 and ctx["player"]["san"] == 21)

# Test 9 — 不该看的日志(M1)
ctx = make_ctx(p_san=20); r = CARDS[9]["effect"](ctx)
check("不该看的日志:foresight+2,SAN-1",
      ctx["player"]["foresight_turns"] == 2 and ctx["player"]["san"] == 19)

# Test 10 — 拆解(M2)
ctx = make_ctx(hand=[3, 4, 7])
ctx["player"]["hand"] = [3, 4, 7]
r = CARDS[10]["effect"](ctx)
check(f"拆解:消耗 1 手牌(剩 {len(ctx['player']['hand'])})+ 10 伤",
      len(ctx["player"]["hand"]) == 2 and r["damage_dealt"] == 10)

# Test 11 — 格式化记忆(M3)
ctx = make_ctx()
ctx["player"]["discard"] = [0, 0, 1]
r = CARDS[11]["effect"](ctx)
check(f"格式化记忆:从 discard 移除 1 张(剩 {len(ctx['player']['discard'])})",
      len(ctx["player"]["discard"]) == 2)

# Test 12 — 亵渎协议(M5)
ctx = make_ctx(); r = CARDS[12]["effect"](ctx)
check("亵渎协议:rule_penalty_multiplier=2,8 伤",
      ctx["player"]["rule_penalty_multiplier"] == 2 and r["damage_dealt"] == 8)

# Test 13 — 无名之吻(M7)
ctx = make_ctx(); r = CARDS[13]["effect"](ctx)
check("无名之吻:20 伤 + SAN-3",
      r["damage_dealt"] == 20 and ctx["player"]["san"] == 27)

# Test 14 — 黑色低语(M7)
ctx = make_ctx(p_san=8); r = CARDS[14]["effect"](ctx)
check("黑色低语:敌人虚弱+3,玩家 SAN+5",
      ctx["enemy"]["weak"] == 3 and ctx["player"]["san"] == 13)

# Test 15
loot = get_loot_pool()
check(f"loot pool 不含疯狂卡(13/14)({len(loot)} 张)",
      13 not in loot and 14 not in loot)

# Test 16
mad = get_madness_cards()
check("madness pool = [13, 14]", set(mad) == {13, 14})

print(f"\n{'🎉' if passed == total else '⚠️'} {passed}/{total} 测试通过")
