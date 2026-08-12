"""env.py — MiniWhisperEnv (v4:cards.py 重写后,信任 effect 返回的 result)"""
import random
import inspect
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cards import (
    CARDS, NUM_CARDS, get_starter_cards, get_loot_pool, get_madness_cards,
)
from enemies import (
    ENEMIES, NUM_ENEMIES, make_enemy, roll_intent, get_enemy_pool,
)
from rules import (
    RULES, NUM_RULES, roll_rules_for_stage,
    check_play_card_violation, check_draw_violation, check_turn_end_violation,
)

MAX_HP = 50
MAX_SAN = 30
MAX_ENERGY = 3
HAND_LIMIT = 10
MAX_FLOORS = 10
DRAW_PER_TURN = 5

ACTION_END_TURN = NUM_CARDS
NUM_ACTIONS = NUM_CARDS + 1


class MiniWhisperEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None, seed=None, max_floors=MAX_FLOORS):
        super().__init__()
        self.render_mode = render_mode
        self.max_floors = max_floors
        intent_dim = 5
        obs_dim = 6 + 5 + intent_dim + 1 + intent_dim + NUM_CARDS + 2 + 5 + NUM_RULES + 2
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-1, high=100, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)
        self.player = {
            "hp": MAX_HP, "max_hp": MAX_HP,
            "san": MAX_SAN, "max_san": MAX_SAN,
            "energy": MAX_ENERGY, "max_energy": MAX_ENERGY,
            "armor": 0, "strength": 0, "weak": 0, "corrosion": 0,
            "overclock_buff": 0, "foresight_turns": 0,
            "rule_penalty_multiplier": 1,
            "deck": list(get_starter_cards()),
            "hand": [], "discard": [], "exhaust": [],
            "buffs": {}, "debuffs": {},
        }
        self.rng.shuffle(self.player["deck"])
        self.floor = 1
        self.turn = 0
        self.played_this_turn = []
        self.done = False
        self.truncated = False
        self.victory = False
        self._start_floor()
        return self._build_obs(), self._info()

    def step(self, action):
        if self.done:
            return self._build_obs(), 0.0, True, self.truncated, self._info()
        reward = 0.0
        info = {}
        if action == ACTION_END_TURN:
            reward += self._end_turn()
        else:
            valid_mask = self.action_masks()
            if not valid_mask[action]:
                reward -= 0.1
                info["invalid_action"] = True
            else:
                reward += self._play_card(action)
        if self.enemy["hp"] <= 0:
            reward += 3.0
            if self.floor >= self.max_floors:
                self.done = True
                self.victory = True
                reward += 20.0
            else:
                self.floor += 1
                reward += self._post_combat_reward()
                self._start_floor()
        elif self.player["hp"] <= 0 or self.player["san"] <= 0:
            self.done = True
            reward -= 5.0
        final_info = self._info()
        final_info.update(info)
        return self._build_obs(), reward, self.done, self.truncated, final_info

    def action_masks(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ACTION_END_TURN] = True
        if self.done: return mask
        for card_id in range(NUM_CARDS):
            if not self._has_card_in_hand(card_id): continue
            card = CARDS[card_id]
            if self.player["energy"] < card["cost"]: continue
            if card.get("madness") and self.player["san"] > 10: continue
            mask[card_id] = True
        return mask

    def _start_floor(self):
        self.turn = 0
        if self.floor <= 3: stage = 0
        elif self.floor <= 6: stage = 1
        elif self.floor <= 9: stage = 2
        else: stage = 3
        pool = get_enemy_pool(stage)
        eid = self.rng.choice(pool)
        self.enemy = make_enemy(eid, rng=self.rng)
        self.enemy.setdefault("buffs", {})
        self.enemy.setdefault("debuffs", {})
        roll_intent(self.enemy)
        self.active_rules = roll_rules_for_stage(stage, rng=self.rng)
        self.player["deck"] = (
            list(self.player["deck"])
            + list(self.player["discard"])
            + list(self.player["hand"])
            + list(self.player.get("exhaust", []))
        )
        self.player["discard"] = []
        self.player["hand"] = []
        self.player["exhaust"] = []
        self.rng.shuffle(self.player["deck"])
        self._start_turn()

    def _start_turn(self):
        self.turn += 1
        self.player["energy"] = self.player["max_energy"]
        self.player["armor"] = 0
        self.player["overclock_buff"] = 0
        self.player["rule_penalty_multiplier"] = 1
        self.played_this_turn = []
        self._draw(DRAW_PER_TURN)

        if self.player["san"] <= 10:
            if self.rng.random() < 0.3:
               madness_pool = get_madness_cards()
               if madness_pool and len(self.player["hand"]) < HAND_LIMIT:
                   madness_id = self.rng.choice(madness_pool)
                   self.player["hand"].append(madness_id)

    def _draw(self, n):
        drawn = 0
        for _ in range(n):
            if not self.player["deck"]:
                if not self.player["discard"]: break
                self.player["deck"] = list(self.player["discard"])
                self.player["discard"] = []
                self.rng.shuffle(self.player["deck"])
            if len(self.player["hand"]) >= HAND_LIMIT: break
            self.player["hand"].append(self.player["deck"].pop())
            drawn += 1
        san_loss = check_draw_violation(self.active_rules, drawn, self.player)
        if san_loss: self.player["san"] -= san_loss

    def _has_card_in_hand(self, card_id):
        return card_id in self.player["hand"]

    def _play_card(self, card_id):
        reward = 0.0
        card = CARDS[card_id]
        self.player["energy"] -= card["cost"]
        self.player["hand"].remove(card_id)

        # 规则检查(带 id)
        card_view = dict(card); card_view["id"] = card_id
        san_pen = check_play_card_violation(self.active_rules, card_view, self.player)
        if san_pen:
            self.player["san"] -= san_pen
            reward -= san_pen * 0.1

        # 调 effect(统一 effect(ctx) 协议)
        ctx = {
            "player": self.player, "enemy": self.enemy, "rng": self.rng,
            "deck": self.player["deck"], "discard": self.player["discard"],
            "exhaust": self.player["exhaust"], "hand": self.player["hand"],
        }
        result = card["effect"](ctx)
        self.played_this_turn.append({"id": card_id, "type": card.get("type")})

        # dest 处理(M3 格式化记忆已经在 effect 内移除目标卡,但卡牌本身仍走 dest)
        dest = card.get("dest", "discard")
        if dest == "discard":
            self.player["discard"].append(card_id)
        elif dest == "exhaust":
            self.player["exhaust"].append(card_id)

        # ⭐ reward shaping(信任 result)
        if isinstance(result, dict):
            reward += 0.05 * result.get("damage_dealt", 0)
            reward += 0.02 * result.get("armor_gained", 0)
            reward += 0.05 * result.get("heal", 0)
            reward += 0.03 * result.get("san_gain", 0)
        return reward

    def _enemy_turn(self):
        reward = 0.0
        intent = self.enemy["next_intent"]
        if intent is None:
            roll_intent(self.enemy); intent = self.enemy["next_intent"]
        itype = intent["type"]
        if itype in ("attack", "big_attack"):
            base = intent["value"] + self.enemy.get("strength", 0)
            if self.enemy.get("weak", 0) > 0: base = int(base * 0.75)
            for _ in range(intent.get("hits", 1)):
                dmg = max(0, base - self.player["armor"])
                self.player["armor"] = max(0, self.player["armor"] - base)
                self.player["hp"] -= dmg
                reward -= 0.05 * dmg
        elif itype == "attack_buff":
            self.enemy["strength"] += intent.get("buff", 1)
            base = intent["value"] + self.enemy["strength"]
            dmg = max(0, base - self.player["armor"])
            self.player["armor"] = max(0, self.player["armor"] - base)
            self.player["hp"] -= dmg
            reward -= 0.05 * dmg
        elif itype == "defend":
            self.enemy["armor"] += intent["value"]
        elif itype == "debuff":
            db = intent.get("debuff"); v = intent["value"]
            if db == "corrosion":   self.player["corrosion"] += v
            elif db == "san_loss":  self.player["san"] -= v; reward -= 0.05 * v
            elif db == "weak":      self.player["weak"] += v
        if self.enemy.get("weak", 0) > 0: self.enemy["weak"] -= 1
        roll_intent(self.enemy)
        return reward

    def _end_turn(self):
        reward = 0.0
        san_pen = check_turn_end_violation(self.active_rules, self.player, self.played_this_turn)
        if san_pen:
            self.player["san"] -= san_pen
            reward -= san_pen * 0.1
        if self.player["corrosion"] > 0:
            dmg = self.player["corrosion"]
            real = max(0, dmg - self.player["armor"])
            self.player["armor"] = max(0, self.player["armor"] - dmg)
            self.player["hp"] -= real
            reward -= 0.05 * real
        self.player["discard"].extend(self.player["hand"])
        self.player["hand"] = []
        if self.enemy["hp"] > 0:
            reward += self._enemy_turn()
        if self.player["foresight_turns"] > 0: self.player["foresight_turns"] -= 1
        if self.player["weak"] > 0: self.player["weak"] -= 1
        if self.player["hp"] > 0 and self.player["san"] > 0 and self.enemy["hp"] > 0:
            self._start_turn()
        return reward

    def _post_combat_reward(self):
        self.player["hp"] = min(self.player["max_hp"], self.player["hp"] + 5)
        self.player["san"] = min(self.player["max_san"], self.player["san"] + 2)
        if self.floor + 1 == self.max_floors:  # 下一层就是 boss
            self.player["hp"] = min(self.player["max_hp"], self.player["hp"] + 15)
            self.player["san"] = min(self.player["max_san"], self.player["san"] + 10)
        loot_pool = get_loot_pool()
        if loot_pool: self.player["deck"].append(self.rng.choice(loot_pool))
        return 0.0

    def _intent_one_hot(self, intent):
        types = ["attack", "attack_buff", "defend", "debuff", "big_attack"]
        oh = [0.0] * 5
        if intent is not None and intent.get("type") in types:
            oh[types.index(intent["type"])] = 1.0
        return oh

    def _build_obs(self):
        p, e = self.player, self.enemy
        obs = [
            p["hp"], p["max_hp"], p["san"], p["max_san"], p["energy"], p["armor"],
            e["hp"], e["max_hp"], e["armor"], e.get("strength", 0), e.get("weak", 0),
        ]
        obs.extend(self._intent_one_hot(e.get("next_intent")))
        obs.append(e.get("next_intent", {}).get("value", 0) if e.get("next_intent") else 0)
        next2 = [0.0] * 5
        if p["foresight_turns"] > 0 and e.get("next_intent"):
            next2 = self._intent_one_hot(e["next_intent"])
        obs.extend(next2)
        hand_count = [0] * NUM_CARDS
        for cid in p["hand"]:
            hand_count[cid] = min(5, hand_count[cid] + 1)
        obs.extend(hand_count)
        obs.append(len(p["deck"]))
        obs.append(len(p["discard"]))
        obs.extend([
            p["corrosion"], p["weak"], p["overclock_buff"],
            p["foresight_turns"], p["rule_penalty_multiplier"],
        ])
        rules_oh = [0.0] * NUM_RULES
        for r in self.active_rules: rules_oh[r] = 1.0
        obs.extend(rules_oh)
        obs.append(self.floor)
        obs.append(self.turn)
        return np.array(obs, dtype=np.float32)

    def _info(self):
        return {
            "floor": self.floor, "turn": self.turn,
            "victory": self.victory,
            "hp": self.player["hp"] if hasattr(self, "player") else 0,
            "san": self.player["san"] if hasattr(self, "player") else 0,
            "active_rules": list(self.active_rules) if hasattr(self, "active_rules") else [],
        }

    def render(self):
        if self.render_mode in ("ansi", "human"):
            return self._render_ansi()

    def _render_ansi(self):
        p, e = self.player, self.enemy
        lines = [
            f"=== 楼层 {self.floor} | 回合 {self.turn} ===",
            f"玩家 HP {p['hp']}/{p['max_hp']}  SAN {p['san']}/{p['max_san']}  能量 {p['energy']}  护甲 {p['armor']}",
            f"  buff: 腐蚀 {p['corrosion']} | 虚弱 {p['weak']} | 预见 {p['foresight_turns']} | M5x{p['rule_penalty_multiplier']}",
            f"敌人 [{e['name']}] HP {e['hp']}/{e['max_hp']}  护甲 {e['armor']}  力量 {e.get('strength',0)}",
        ]
        if e.get("next_intent"):
            lines.append(f"  意图: {e['next_intent'].get('desc','?')}")
        lines.append(f"规则: {self.active_rules}")
        hand_str = ", ".join(f"[{c}]{CARDS[c]['name']}({CARDS[c]['cost']})" for c in p["hand"])
        lines.append(f"手牌: {hand_str}")
        lines.append(f"抽 {len(p['deck'])} | 弃 {len(p['discard'])}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text
