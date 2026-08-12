"""rules.py v2 测试 - 适配阈值"""
import random
from rules import (
    RULES, NUM_RULES, roll_rules_for_stage,
    check_play_card_violation, check_draw_violation,
    check_turn_end_violation, get_rule_descriptions,
)

passed = 0
total = 0

def check(name, cond):
    global passed, total
    total += 1
    print(f"{'✅' if cond else '❌'} Test {total}: {name}")
    if cond: passed += 1


# Test 1
check("规则数量为 5", NUM_RULES == 5)

# Test 2
check("所有规则有 name/desc/trigger 字段",
      all("name" in r and "desc" in r and "trigger" in r for r in RULES.values()))

# Test 3
check("普通层返回 1 条规则", len(roll_rules_for_stage(0, random.Random(42))) == 1)

# Test 4
check("Boss 层返回 1 条规则", len(roll_rules_for_stage(3, random.Random(42))) == 1)
check("Stage 2 也是单规则", len(roll_rules_for_stage(2, random.Random(42))) == 1)

# Test 5 - 规则 0:禁用「不该看的日志」
player = {"rule_penalty_multiplier": 1}
forbidden = {"id": 9, "type": "skill"}
check("打出禁用卡 -> 扣 3 SAN", check_play_card_violation([0], forbidden, player) == 3)

# Test 6
ok = {"id": 0, "type": "attack"}
check("打出未禁用的卡 -> 不扣", check_play_card_violation([0], ok, player) == 0)

# Test 7 - M5 加倍
player_m5 = {"rule_penalty_multiplier": 2}
check("亵渎协议激活时扣 6 SAN(2倍)", check_play_card_violation([0], forbidden, player_m5) == 6)

# Test 8 - 规则 1:抽 5 张不扣(在阈值内)
check("抽 5 张(阈值内)-> 不扣", check_draw_violation([1], 5, player) == 0)

# Test 9 - 规则 1:抽 7 张扣 (7-5)*1 = 2
check("抽 7 张(超 2)-> 扣 2 SAN", check_draw_violation([1], 7, player) == 2)

# Test 10 - 规则 3:攻击牌
attack = {"id": 0, "type": "attack"}
check("禁攻击层打攻击牌 -> 扣 2 SAN", check_play_card_violation([3], attack, player) == 2)

# Test 11 - 规则 2:手牌 2 张(在阈值内)-> 不扣
p_hand_2 = {"rule_penalty_multiplier": 1, "hand": [{"id": 0}, {"id": 1}]}
check("手牌 2 张(阈值内)-> 不扣", check_turn_end_violation([2], p_hand_2, []) == 0)

# Test 12 - 规则 2:手牌 4 张 -> (4-2)*2 = 4 SAN
p_hand_4 = {"rule_penalty_multiplier": 1, "hand": [{"id": 0}]*4}
check("手牌 4 张 -> 扣 4 SAN", check_turn_end_violation([2], p_hand_4, []) == 4)

# Test 13 - 规则 4:没打技能 -> 罚
p_empty = {"rule_penalty_multiplier": 1, "hand": []}
check("规则 4:没打技能/能力牌 -> 扣 3 SAN",
      check_turn_end_violation([4], p_empty, [{"type": "attack"}]) == 3)

# Test 14 - 规则 4:打了技能 -> 不扣
check("规则 4:打了技能牌 -> 不扣",
      check_turn_end_violation([4], p_empty, [{"type": "skill"}]) == 0)

# Test 15 - 规则 4:打了能力牌 -> 不扣
check("规则 4:打了能力牌 -> 不扣",
      check_turn_end_violation([4], p_empty, [{"type": "power"}]) == 0)

# Test 16 - 规则叠加
p_multi = {"rule_penalty_multiplier": 1, "hand": [{"id": 0}]*4}
# 规则 2: (4-2)*2 = 4 ; 规则 4: 没打技能 = 3 ; 总 7
check("两条规则同时触发 -> 扣 7 SAN",
      check_turn_end_violation([2, 4], p_multi, [{"type": "attack"}]) == 7)

# Test 17
check("get_rule_descriptions 返回字符串列表",
      len(get_rule_descriptions([0, 1])) == 2)

print(f"\n{'🎉' if passed == total else '⚠️'} {passed}/{total} 测试通过")
