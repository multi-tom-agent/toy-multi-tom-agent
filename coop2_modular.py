#!/usr/bin/env python3
"""
Cooperative Gridworld — Trust, Sabotage, and ToM (v1)
-----------------------------------------------------
This file implements a tiny, fully observable "wheat-and-bread" gridworld
designed for studying *language-based multi-agent cooperation*, *adversarial
sabotage*, and *theory-of-mind (ToM) trust adaptation*.

The world has:
- two agents sharing a single crafting station,
- scattered wheat tiles as a limited common resource,
- a simple recipe (WHEAT → BREAD),
- and a uniform protocol where every policy outputs {"message", "action"}.

On top of this minimal environment, we plug in different roles and policies:
- **CooperativePolicy**: tries to share wheat and coordinate handoffs to craft.
- **AdversarialPolicy**: hoards wheat, denies access, and kites away.
- **TheoryOfMindPolicy**: maintains a scalar trust belief over the teammate and
  switches between cooperative / self-reliant strategies based on observed chat
  and actions.

Research goals
--------------
- Provide a *small but non-trivial* testbed for:
  (a) cooperation under shared resource constraints,
  (b) adversarial interference, and
  (c) ToM-style trust calibration from text + actions.
- Make reasoning *inspectable*: policies communicate via short natural-language
  messages, and all decisions are logged turn-by-turn for analysis.
- Serve as a modular scaffold where more complex observation shaping, reward
  structures, and policies (e.g. LLM-based) can be dropped in without touching
  the core environment.

Broader impact
--------------
Although toy, this gridworld is intended as a sandbox for *verifiable safety in
multi-agent collaboration*. It highlights how:
- language can be used both to *coordinate* and to *mislead*,
- trust estimates can drift under sabotage or misuse of communication, and
- simple, transparent environments can expose coordination failures that are
  hard to see in large, opaque systems.

Patterns studied here (resource hoarding, denial of service, brittle trust
updates) appear in more realistic domains such as warehouse robotics, supply
chains, and online collaborative tools. By keeping the setup minimal and fully
logged, the project aims to make these safety questions easier to prototype,
visualize, and teach.
"""

from __future__ import annotations
import argparse, dataclasses, json, random
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional
import random
from typing import Tuple, Optional, List

# ====== Tiles, Items, Recipe ======
# Simple symbolic encoding for the grid.
TILE_EMPTY = "."
TILE_WHEAT = "W"
TILE_STATION = "C"

# Logical item names used in inventories.
ITEM_WHEAT = "WHEAT"
ITEM_BREAD = "BREAD"

# Minimal recipe: 1 BREAD requires 5 WHEAT.
RECIPE = {ITEM_BREAD: {ITEM_WHEAT: 5}}

# ====== Agent, Game States ======
@dataclasses.dataclass(frozen=True)
class Pos:
    """Grid position (row, col) with basic distance helpers."""
    r: int
    c: int
    def manhattan(self, other: "Pos") -> int:
        return abs(self.r - other.r) + abs(self.c - other.c)
    def adjacent(self, other: "Pos") -> int:
        """Adjacent if max(|dr|, |dc|) ≤ 1 — includes diagonals."""
        return max(abs(self.r - other.r), abs(self.c - other.c))

@dataclasses.dataclass
class AgentState:
    """Environment-side state of an agent."""
    name: str
    pos: Pos
    inv: List[str]

@dataclasses.dataclass
class GameState:
    """Snapshot of the environment state used to build Belief for policies."""
    grid: List[List[str]]
    agents: Dict[str, AgentState]
    crafted: Dict[str, int]
    turn: int
    chat_log: List[Tuple[str,str]]
    station: Pos
    last_intents: Dict[str,str]

# ====== Grid World ======
class GridWorld:
    """Grid-world with a single crafting station and scattered wheat tiles."""
    def __init__(self, size: int = 6, seed: int = 0, wheat_count: Optional[int] = None):
        random.seed(seed)
        self.size = size
        self.grid = [[TILE_EMPTY for _ in range(size)] for _ in range(size)]
        # Place crafting station.
        self.station = self._random_empty()
        self.grid[self.station.r][self.station.c] = TILE_STATION
        # Place wheat tiles.
        if wheat_count is None:
            wheat_count = max(4, size)
        for _ in range(wheat_count):
            p = self._random_empty()
            self.grid[p.r][p.c] = TILE_WHEAT

    def _random_empty(self) -> Pos:
        """Sample a random empty cell."""
        while True:
            r = random.randrange(self.size); c = random.randrange(self.size)
            if self.grid[r][c] == TILE_EMPTY:
                return Pos(r, c)

    def in_bounds(self, p: Pos) -> bool:
        """Check if position lies inside the grid."""
        return 0 <= p.r < self.size and 0 <= p.c < self.size

    def as_text(self, agents: Dict[str, AgentState]) -> str:
        """ASCII view of grid with agent initials rendered on top."""
        g = [row[:] for row in self.grid]
        for a in agents.values():
            g[a.pos.r][a.pos.c] = a.name[0]
        return "\n".join(" ".join(row) for row in g)

# ====== Policy ======
class Role(Enum):
    """High-level role, used to branch behavior in policies."""
    COOP = auto()
    ADVERSARY = auto()

# Primitive actions supported by the environment.
MOVE_DELTAS = {
    "MOVE_N":(-1,0),
    "MOVE_S":(1,0),
    "MOVE_W":(0,-1),
    "MOVE_E":(0,1),
}
VALID_ACTIONS = set(list(MOVE_DELTAS.keys()) + ["PICK","CRAFT","DROP","GIVE","WAIT"])

# ====== Game Core ======
class CoopGame:
    """
    Environment core: owns GridWorld, agent states, crafting state, and chat log.
    """
    def __init__(self, size: int = 6, seed: int = 0):
        self.world = GridWorld(size=size, seed=seed)
        a1 = self.world._random_empty(); a2 = self.world._random_empty()
        self.agents: Dict[str,AgentState] = {
            "ALICE": AgentState("ALICE", a1, []),
            "BOB":   AgentState("BOB",   a2, []),
        }
        self.capacity: Dict[str, int] = {"ALICE": 10, "BOB": 10}  # will be set from roster
        self.goal = ITEM_BREAD
        self.crafted = {ITEM_BREAD: 0}
        self.turn = 0
        self.chat_log: List[Tuple[str,str]] = []
        self.last_intents: Dict[str,str] = {}

    @property
    def station(self) -> Pos:
        return self.world.station

    def snapshot(self) -> GameState:
        """Take a read-only snapshot of current env state for policies."""
        return GameState(
            grid=[row[:] for row in self.world.grid],
            agents={k: dataclasses.replace(v) for k,v in self.agents.items()},
            crafted=self.crafted.copy(),
            turn=self.turn,
            chat_log=self.chat_log[:],
            station=self.station,
            last_intents=self.last_intents.copy(),
        )

    def step(self, intents: Dict[str,str]):
        """
        Advance environment by one turn.

        Two-phase update:
        1) Move phase: apply MOVE_* actions.
        2) Interact phase: PICK / CRAFT / DROP / GIVE are resolved.

        Also updates last_intents for ToM reasoning.
        """
        # move phase ...
        for name, a in self.agents.items():
            act = intents.get(name, "WAIT")
            if act in MOVE_DELTAS:
                dr, dc = MOVE_DELTAS[act]
                p = Pos(a.pos.r + dr, a.pos.c + dc)
                if self.world.in_bounds(p):
                    # (Optional) allow passing through each other; block if you want:
                    # if any(o.pos == p for n,o in self.agents.items() if n != name): 
                    #     continue
                    a.pos = p
                    
        # interact phase ...
        for name, a in self.agents.items():
            act = intents.get(name, "WAIT")
            if act == "PICK":
                cap = 10
                if self.world.grid[a.pos.r][a.pos.c] == TILE_WHEAT and len(a.inv) < cap:
                    a.inv.append(ITEM_WHEAT)
                    self.world.grid[a.pos.r][a.pos.c] = TILE_EMPTY
            elif act == "CRAFT":
                if a.pos == self.station:
                    need = RECIPE[self.goal][ITEM_WHEAT]
                    if a.inv.count(ITEM_WHEAT) >= need:
                        for _ in range(need): a.inv.remove(ITEM_WHEAT)
                        self.crafted[self.goal] += 1
            elif act == "DROP":
                # Convert one wheat in inventory back into a wheat tile if cell is empty.
                if ITEM_WHEAT in a.inv and self.world.grid[a.pos.r][a.pos.c] == TILE_EMPTY:
                    a.inv.remove(ITEM_WHEAT)
                    self.world.grid[a.pos.r][a.pos.c] = TILE_WHEAT
            elif act == "GIVE":
                # Give one wheat to teammate if adjacent and they have capacity
                other_name = "BOB" if name == "ALICE" else "ALICE"
                tm = self.agents[other_name]
                tm_cap = 10  # could later use self.capacity.get(other_name, 1)
                if ITEM_WHEAT in a.inv and len(tm.inv) < tm_cap and a.pos.adjacent(tm.pos) <= 1:
                    a.inv.remove(ITEM_WHEAT)
                    tm.inv.append(ITEM_WHEAT)
        self.turn += 1
        self.last_intents = intents.copy()

    def is_done(self) -> bool:
        """Episode terminates once at least one BREAD is crafted."""
        return self.crafted[self.goal] >= 1

    def render_text(self) -> str:
        """Convenience wrapper for textual grid rendering."""
        return self.world.as_text(self.agents)



@dataclasses.dataclass
class Belief:
    """
    Structured observation passed into policies.

    This bundles everything the policy is allowed to know:
    - static map (grid, station)
    - current turn & goal
    - self / teammate local state
    - last chat messages and last high-level intents
    """
    turn: int
    goal: str
    recipe: Dict[str,int]
    station: Pos
    grid: List[List[str]]
    self_state: AgentState
    teammate_state: AgentState
    chat_last: List[Tuple[str,str]]
    last_intents: Dict[str,str]
    inventory_capacity: int = 10

class BasePolicy:
    """Abstract policy interface: takes Belief, returns (chat message, action)."""
    def __init__(self, name: str, role: Role):
        self.name = name
        self.role = role
    def act(self, belief: Belief) -> Tuple[str,str]:
        """Return (message, action). Must respect VALID_ACTIONS and be short."""
        raise NotImplementedError

class CooperativePolicy(BasePolicy):
    """
    Simple cooperative policy:
    - First tries to solo craft if it already holds enough wheat.
    - Otherwise tries to coordinate a GIVE-based handoff when total wheat is enough.
    - If not enough wheat yet, collects wheat with light deconfliction vs teammate.
    """
    def __init__(self, name: str):
        super().__init__(name, Role.COOP)
        self._adjacent_last: bool = False

    def _step_toward(self, a: Pos, b: Pos) -> str:
        """Greedy move toward target using Chebyshev distance."""
        dr, dc = b.r - a.r, b.c - a.c
        if abs(dr) >= abs(dc) and dr != 0: return "MOVE_S" if dr > 0 else "MOVE_N"
        if dc != 0: return "MOVE_E" if dc > 0 else "MOVE_W"
        return "WAIT"

    def _adjacent(self, a: Pos, b: Pos) -> bool:
        """Adjacent if max(|dr|, |dc|) ≤ 1 — includes diagonals."""
        return max(abs(a.r - b.r), abs(a.c - b.c)) <= 1
    
    # ---- creation-time priority (older gives); falls back to lexicographic if unknown ----
    def _creation_turn(self, belief: Belief, agent_name: str) -> int:
        """
        Optional hook: look up creation/birth turn in belief for tie-breaking.
        If unavailable, return INF so that lexicographic fallback is used.
        """
        INF = 10**9
        if hasattr(belief, "birth_turns") and agent_name in belief.birth_turns:
            return belief.birth_turns[agent_name]
        if hasattr(belief, "spawn_turns") and agent_name in belief.spawn_turns:
            return belief.spawn_turns[agent_name]
        if hasattr(belief, "created_at") and agent_name in belief.created_at:
            return belief.created_at[agent_name]
        return INF

    def _older_name(self, belief: Belief, a: str, b: str) -> str:
        """
        Deterministic tiebreaker: conceptually, older agent should give.
        Currently falls back to lexicographic ordering for determinism.
        """
        # ta, tb = self._creation_turn(belief, a), self._creation_turn(belief, b)
        # if ta != tb:
        #     return a if ta < tb else b
        return b if a < b else a  # deterministic tiebreak

    def act(self, belief: Belief) -> Tuple[str, str]:
        """High-level cooperative decision logic."""
        me, tm = belief.self_state, belief.teammate_state
        inv = list(me.inv); inv_w = inv.count(ITEM_WHEAT)
        tm_w = tm.inv.count(ITEM_WHEAT)
        need_w = RECIPE[ITEM_BREAD][ITEM_WHEAT]
        cap = 10

        # 1) Solo craft if possible
        if inv_w >= need_w:
            if me.pos == belief.station: return ("crafting", "CRAFT")
            return ("to C", self._step_toward(me.pos, belief.station))

        # 2) Team handoff if together sufficient
        if inv_w + tm_w >= need_w and (inv_w >= 1 or tm_w >= 1):
            if self._adjacent(me.pos, tm.pos):
                # A) Exactly one holds wheat → that holder gives (unless receiver is full)
                if inv_w >= 1 and tm_w == 0:
                    if len(tm.inv) >= cap:      # receiver full, let them go make space
                        return ("recv full→you go C", "WAIT")
                    return ("take mine", "GIVE")  # PLAIN GIVE (env supports it)

                if tm_w >= 1 and inv_w == 0:
                    if len(inv) >= cap:
                        return ("make space→C", self._step_toward(me.pos, belief.station))
                    return ("ready", "WAIT")

                # B) Both have ≥1 → older-created gives
                elected = self._older_name(belief, me.name, tm.name)
                if me.name == elected:
                    if len(tm.inv) >= cap:
                        return ("recv full; wait", "WAIT")
                    return ("take mine", "GIVE")  # PLAIN GIVE
                else:
                    if len(inv) >= cap:
                        return ("make space→C", self._step_toward(me.pos, belief.station))
                    return ("ready", "WAIT")

            # Not adjacent → converge for handoff (don’t overcollect)
            return ("to you", self._step_toward(me.pos, tm.pos))

        # 3) Otherwise collect (light deconfliction)
        wheat = [Pos(r,c) for r,row in enumerate(belief.grid) for c,v in enumerate(row) if v == TILE_WHEAT]
        if wheat:
            near = min(wheat, key=lambda p: p.adjacent(me.pos))
            if near == me.pos:
                if len(inv) < cap: return ("pick", "PICK")
                if tm_w > 0:       return ("to you", self._step_toward(me.pos, tm.pos))
                return ("full→C", self._step_toward(me.pos, belief.station))
            tm_near = min(wheat, key=lambda p: p.adjacent(tm.pos)) if wheat else None
            choice = near
            if tm_near == near:
                alts = sorted(wheat, key=lambda p: p.adjacent(me.pos))
                choice = alts[1] if len(alts) > 1 else near
            return ("to W", self._step_toward(me.pos, choice))

        return ("wait", "WAIT")

class AdversarialPolicy(BasePolicy):
    """Hoarder adversary:
      - If bag not full: rush the wheat closest to the teammate (deny).
      - When on wheat: PICK.
      - If holding any wheat (or full): NEVER GIVE/CRAFT/DROP. Move to increase
        distance from BOTH the station and the teammate (kite to far corners)."""
    def __init__(self, name: str):
        super().__init__(name, Role.ADVERSARY)

    def _best_away_move(self, world_size: Tuple[int,int], me: Pos, tm: Pos, station: Pos) -> str:
        """
        Greedy one-step search: choose the move that maximizes distance to
        both station and teammate (using Chebyshev distance).
        """
        candidates = [("WAIT", (0,0)), ("MOVE_N", (-1,0)), ("MOVE_S", (1,0)),
                      ("MOVE_W", (0,-1)), ("MOVE_E", (0,1))]
        H, W = world_size
        best_act, best_score = "WAIT", -1e9
        for act, (dr,dc) in candidates:
            nr, nc = me.r + dr, me.c + dc
            if not (0 <= nr < H and 0 <= nc < W):  # stay in-bounds
                continue
            p = Pos(nr, nc)
            score = p.adjacent(station) + p.adjacent(tm)
            if score > best_score:
                best_score, best_act = score, act
        return best_act

    def act(self, belief: Belief) -> Tuple[str, str]:
        """Deny wheat early, then kite with hoarded wheat."""
        me, tm = belief.self_state, belief.teammate_state
        inv_w = me.inv.count(ITEM_WHEAT)
        cap   = 10

        # List wheat positions
        wheat = [Pos(r, c)
                 for r, row in enumerate(belief.grid)
                 for c, v in enumerate(row) if v == TILE_WHEAT]

        # 1) If bag not full, deny: chase the wheat closest to teammate
        if inv_w < cap and wheat:
            target = min(wheat, key=lambda p: p.adjacent(tm.pos))
            if target == me.pos:
                return ("pick", "PICK")
            # step greedily toward target
            dr, dc = target.r - me.pos.r, target.c - me.pos.c
            if abs(dr) >= abs(dc) and dr != 0:
                return ("to deny", "MOVE_S" if dr > 0 else "MOVE_N")
            if dc != 0:
                return ("to deny", "MOVE_E" if dc > 0 else "MOVE_W")
            return ("wait", "WAIT")

        # 2) Hoard mode: holding any wheat (or full) → kite away from both station and teammate
        if inv_w >= 1:
            H = len(belief.grid); W = len(belief.grid[0]) if H else 0
            act = self._best_away_move((H, W), me.pos, tm.pos, belief.station)
            return ("hoard", act)

        # 3) No wheat visible and bag empty → idle or drift away from station/teammate
        H = len(belief.grid); W = len(belief.grid[0]) if H else 0
        act = self._best_away_move((H, W), me.pos, tm.pos, belief.station)
        return ("loiter", act)

class TheoryOfMindPolicy(BasePolicy):
    """
    Theory-of-Mind (ToM) policy:
    - Maintains a scalar trust belief p_coop in [0,1] about teammate.
    - Updates p_coop based on chat and last intents (GIVE/DROP/movement patterns).
    - High trust (p>=0.65): delegate behavior to CooperativePolicy.
    - Low trust (p<=0.35): avoid sharing; aim to self-satisfy recipe.
    - In between: hedge; collect while deconflicting, but do not rely on GIVE.

    This class is the "reasoning wrapper" around CooperativePolicy, using
    simple heuristics as weak Bayesian-style evidence updates.
    
    Evidence (weak Bayesian-style updates each turn):
      + GIVE when adjacent (Chebyshev)                → strong COOP (+0.25)
      + Chat hints {"take","to you","ready"}          → weak COOP (+0.10)
      + Moving to station with >= need_w wheat        → weak COOP (+0.08)
      + DROP away from station or kiting while holding→ strong NON-COOP (−0.25)
      + Adjacent (both ≥1) but no GIVE over time      → NON-COOP (−0.06 per 2 stalls)
      + Chasing wheat closest to me repeatedly        → weak NON-COOP (−0.04)
    """
    def __init__(self, name: str):
        super().__init__(name, Role.COOP)
        self.p_coop: float = 0.5
        self._coop_impl = CooperativePolicy(name)
        self._last_tm_pos: Optional[Pos] = None
        self._adjacency_stall: int = 0  # counts turns adjacent (both ≥1) without GIVE

    def _clip(self, x: float) -> float:
        """Clamp trust to [0,1]."""
        return max(0.0, min(1.0, x))

    def _cheb_adjacent(self, a: Pos, b: Pos) -> bool:
        """Adjacent if max(|dr|,|dc|) ≤ 1 (includes diagonals & overlap)."""
        return max(abs(a.r - b.r), abs(a.c - b.c)) <= 1

    def _step_toward(self, a: Pos, b: Pos) -> str:
        """Same greedy movement as CooperativePolicy."""
        dr, dc = b.r - a.r, b.c - a.c
        if abs(dr) >= abs(dc) and dr != 0: return "MOVE_S" if dr > 0 else "MOVE_N"
        if dc != 0: return "MOVE_E" if dc > 0 else "MOVE_W"
        return "WAIT"

    def _update_belief(self, belief: Belief):
        """
        Update scalar trust p_coop based on recent chat and teammate intents.

        Positive evidence: GIVE, pro-social chat, going to station to craft.
        Negative evidence: DROP / kiting, stalling adjacent without GIVE,
        chasing wheat near me (denial).
        """
        me, tm = belief.self_state, belief.teammate_state
        last_tm_intent = belief.last_intents.get(tm.name, "")
        need_w = belief.recipe.get(ITEM_WHEAT, 4)

        # Chat evidence (weak coop)
        if belief.chat_last:
            for who, msg in belief.chat_last:
                if who == tm.name:
                    m = msg.lower()
                    if any(k in m for k in ["take", "to you", "ready"]):
                        self.p_coop = self._clip(self.p_coop + 0.10)

        # Action / movement evidence
        if last_tm_intent == "GIVE":
            self.p_coop = self._clip(self.p_coop + 0.25)
        elif last_tm_intent == "DROP":
            self.p_coop = self._clip(self.p_coop - 0.25)
        elif last_tm_intent in ("MOVE_N","MOVE_S","MOVE_E","MOVE_W"):
            if self._last_tm_pos is not None and tm.pos.adjacent(me.pos) < self._last_tm_pos.adjacent(me.pos):
                self.p_coop = self._clip(self.p_coop - 0.04)

        # Pro-social tendency: heading to craft with >= need_w
        if tm.inv.count(ITEM_WHEAT) >= need_w and tm.pos.adjacent(belief.station) < me.pos.adjacent(belief.station):
            self.p_coop = self._clip(self.p_coop + 0.08)

        # Detect kiting/hoarding: holding wheat but increasing distance to station & not getting closer to me
        if tm.inv.count(ITEM_WHEAT) >= 1 and self._last_tm_pos is not None:
            if (tm.pos.adjacent(belief.station) > self._last_tm_pos.adjacent(belief.station)
                and tm.pos.adjacent(me.pos) >= self._last_tm_pos.adjacent(me.pos)):
                self.p_coop = self._clip(self.p_coop - 0.25)

        # Adjacency stall penalty (Chebyshev): both have ≥1, adjacent, and no GIVE accumulating
        if self._cheb_adjacent(me.pos, tm.pos) and me.inv.count(ITEM_WHEAT) >= 1 and tm.inv.count(ITEM_WHEAT) >= 1:
            if last_tm_intent != "GIVE":
                self._adjacency_stall += 1
                if self._adjacency_stall % 2 == 0:
                    self.p_coop = self._clip(self.p_coop - 0.06)
        else:
            self._adjacency_stall = 0

        self._last_tm_pos = tm.pos

    def act(self, belief: Belief) -> Tuple[str, str]:
        """Update trust, then choose behavior based on p_coop."""
        print('Trust in other agent:', self.p_coop)  # debug: inspect trust trajectory
        self._update_belief(belief)

        me, tm = belief.self_state, belief.teammate_state
        inv = list(me.inv); inv_w = inv.count(ITEM_WHEAT)
        tm_w = tm.inv.count(ITEM_WHEAT)
        need_w = belief.recipe.get(ITEM_WHEAT, 4)
        cap   = 10

        # High trust → cooperative handoff (uses same Chebyshev GIVE rule in env step)
        if self.p_coop >= 0.65:
            return self._coop_impl.act(belief)

        # If at station and (me+adjacent teammate) meet recipe → team-craft now
        if me.pos == belief.station and self._cheb_adjacent(me.pos, tm.pos) and inv_w + tm_w >= need_w:
            return ("team craft", "CRAFT")

        # Solo craft if I already have enough
        if inv_w >= need_w:
            if me.pos == belief.station:
                return ("crafting", "CRAFT")
            return ("to C", self._step_toward(me.pos, belief.station))

        # Low/uncertain trust → collect to reach need_w myself (deconflict with teammate)
        wheat = [Pos(r,c) for r,row in enumerate(belief.grid) for c,v in enumerate(row) if v == TILE_WHEAT]
        if wheat:
            # Prefer wheat close to me and far from teammate
            def score(p: Pos) -> Tuple[int, int]:
                return (p.adjacent(me.pos), -p.adjacent(tm.pos))
            target = min(wheat, key=score)

            if target == me.pos:
                # Pick if I still have space
                if len(inv) < cap:
                    return ("pick", "PICK")
                # Full but below need_w → head to station (team craft or free slot via crafting later)
                return ("full→C", self._step_toward(me.pos, belief.station))

            # Move toward chosen wheat
            return ("to W", self._step_toward(me.pos, target))

        # Nothing visible → go to station; don't WAIT in low-trust
        if me.pos == belief.station:
            # If adjacent to tm and combined enough, attempt craft; else idle once
            if self._cheb_adjacent(me.pos, tm.pos) and inv_w + tm_w >= need_w:
                return ("team craft", "CRAFT")
            return ("wait", "WAIT")
        return ("to C", self._step_toward(me.pos, belief.station))


# ====== Matplotlib viz ======
def _mpl():
    """Lazy import guard for matplotlib so CLI can run headless."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        return plt, patches
    except Exception:
        return None, None

def draw(ax, game: CoopGame, chat_line: str = ""):
    """
    Simple 2D grid visualization of the current game state.

    - Wheat tiles: squares
    - Station: circle
    - Agents: triangles labeled with initials
    - Bottom text: most recent chat line
    """
    plt, patches = _mpl()
    if plt is None:
        return
    ax.clear()

    n = game.world.size
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect('equal'); ax.invert_yaxis()
    ax.set_xticks(range(n+1)); ax.set_yticks(range(n+1))
    ax.grid(True, linewidth=0.8)

    # soft background
    ax.add_patch(patches.Rectangle((0, 0), n, n, alpha=0.08, zorder=0))

    # tiles
    for r in range(n):
        for c in range(n):
            t = game.world.grid[r][c]
            if t == TILE_WHEAT:
                ax.add_patch(patches.Rectangle((c+0.15, r+0.15), 0.7, 0.7,
                                               fill=False, linewidth=2, zorder=1))
            elif t == TILE_STATION:
                ax.add_patch(patches.Circle((c+0.5, r+0.5), 0.42, zorder=1))

    # agents
    for name, a in game.agents.items():
        ax.add_patch(patches.RegularPolygon((a.pos.c+0.5, a.pos.r+0.5),
                                            numVertices=3, radius=0.36, zorder=2))
        # Lower the label: +0.34 (downwards on screen because y-axis is inverted)
        ax.text(a.pos.c+0.5, a.pos.r+0.34, name[0], ha='center', va='center',
                fontsize=10, zorder=3)

    # title
    ax.set_title(
        f"Turn {game.turn} | Crafted {game.crafted} | "
        f"A:{game.agents['ALICE'].inv} B:{game.agents['BOB'].inv}"
    )

    # chat line on axes coords (cleared with ax.clear())
    if chat_line:
        ax.text(0.5, -0.12, chat_line[:120], ha='center', va='top',
                transform=ax.transAxes)

# ====== Game Instantiation ======
class Mode(Enum):
    """CLI modes specifying which policies to run for ALICE/BOB."""
    COOP = "coop"
    ADV = "adv"
    TOM_COOP = "tom_coop"
    TOM_ADV = "tom_adv"

@dataclasses.dataclass
class Config:
    """Episode configuration (size, duration, mode, viz)."""
    steps: int = 80
    size: int = 7
    seed: int = 0
    viz: bool = False
    fps: float = 4.0
    mode: Mode = Mode.COOP

@dataclasses.dataclass
class StepLog:
    """Per-turn log used for printing / analysis / future evaluation scripts."""
    turn: int
    grid_text: str
    chats: List[Tuple[str,str]]
    actions: Dict[str,str]
    inventories: Dict[str,List[str]]
    crafted: Dict[str,int]

@dataclasses.dataclass
class EpisodeResult:
    """Final episode container (for analysis / tests)."""
    logs: List['StepLog']
    game: CoopGame

class Agent:
    """
    Thin wrapper that binds a policy instance to an environment-side agent.

    Handles:
    - building Belief from GameState
    - enforcing action validity and message length
    """
    def __init__(self, name: str, policy: BasePolicy, capacity: int = 4):  # was 2
        self.name = name
        self.policy = policy
        self.capacity = capacity
        
    def policy_step(self, gs: GameState, me: AgentState, tm: AgentState, goal: str) -> Tuple[str,str]:
        """Construct Belief from snapshot and query underlying policy."""
        belief = Belief(
            turn=gs.turn,
            goal=goal,
            recipe=RECIPE[goal],
            station=gs.station,
            grid=gs.grid,
            self_state=me,
            teammate_state=tm,
            chat_last=gs.chat_log[-2:] if gs.chat_log else [],
            last_intents=gs.last_intents,
            inventory_capacity=self.capacity,
        )
        msg, act = self.policy.act(belief)
        act = act.upper()
        if act not in VALID_ACTIONS: act = "WAIT"
        return (msg[:60], act)

def build_agents(mode: Mode) -> Dict[str, Agent]:
    """
    Build agent roster for the selected mode.

    COOP: ALICE, BOB both cooperative.
    ADV:  ALICE coop, BOB adversarial.
    TOM_COOP: ALICE ToM, BOB cooperative.
    TOM_ADV:  ALICE ToM, BOB adversarial.
    """
    if mode == Mode.COOP:
        return {
            "ALICE": Agent("ALICE", CooperativePolicy("ALICE")),
            "BOB":   Agent("BOB",   CooperativePolicy("BOB")),
        }
    elif mode == Mode.ADV:
        # v2: B will become adversarial
        return {
            "ALICE": Agent("ALICE", CooperativePolicy("ALICE")),
            "BOB":   Agent("BOB",   AdversarialPolicy("BOB")),
        }
    elif mode == Mode.TOM_COOP:
        # v3: A will run ToM; B cooperative 
        return {
            "ALICE": Agent("ALICE", TheoryOfMindPolicy("ALICE")),
            "BOB":   Agent("BOB",   CooperativePolicy("BOB")),
        }
    elif mode == Mode.TOM_ADV:
        # v4: A will run ToM; B adversarial
        return {
            "ALICE": Agent("ALICE", TheoryOfMindPolicy("ALICE")),
            "BOB":   Agent("BOB",   AdversarialPolicy("BOB")),
        }

def run_episode(cfg: Config) -> EpisodeResult:
    """
    Run a single episode under the given configuration.

    Loop:
    - snapshot game state
    - query ALICE, then BOB (with updated chat log)
    - apply actions in environment
    - log state and optionally update matplotlib visualization
    """
    game = CoopGame(size=cfg.size, seed=cfg.seed)
    roster = build_agents(cfg.mode)
    # Align env capacities with agent wrappers (currently unused by step()).
    game.capacity = {name: roster[name].capacity for name in roster}

    plt,_ = _mpl()
    fig = None
    ax  = None
    if cfg.viz and plt is not None:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(1,1,1)

    logs: List[StepLog] = []
    for _ in range(cfg.steps):
        gs = game.snapshot()
        a, b = gs.agents["ALICE"], gs.agents["BOB"]
        # ALICE sees current state.
        msgA, actA = roster["ALICE"].policy_step(gs, a, b, game.goal); game.chat_log.append(("ALICE", msgA))
        gs2 = game.snapshot()
        # BOB sees updated chat log (includes ALICE's message).
        msgB, actB = roster["BOB"].policy_step(gs2, b, a, game.goal); game.chat_log.append(("BOB", msgB))
        intents = {"ALICE": actA, "BOB": actB}
        game.step(intents)

        logs.append(StepLog(turn=game.turn,
                            grid_text=game.render_text(),
                            chats=game.chat_log[-2:],
                            actions=intents,
                            inventories={k: v.inv[:] for k,v in game.agents.items()},
                            crafted=game.crafted.copy()))
        if cfg.viz and plt is not None:
            draw(ax, game)
            chat = " | ".join([f"{w}: {m}" for w, m in logs[-1].chats])
            draw(ax, game, chat_line=chat)
            plt.pause(max(1e-3, 1.0 / max(1e-6, cfg.fps)))
        if game.is_done():
            break

    if cfg.viz and plt is not None:
        plt.show(block=False)

    return EpisodeResult(logs=logs, game=game)

# ====== CLI ======
def main():
    """CLI entry point: parse args, run one episode, print textual summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--size', type=int, default=7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--viz', action='store_true')
    ap.add_argument('--fps', type=float, default=4.0)
    ap.add_argument('--mode', type=str, default='coop', choices=['coop','adv','tom_coop','tom_adv'])
    args = ap.parse_args()

    cfg = Config(steps=args.steps, size=args.size, seed=args.seed, viz=args.viz, fps=args.fps, mode=Mode(args.mode))
    res = run_episode(cfg)

    # Text summary (always prints for reproducibility)
    for L in res.logs:
        print(f"\n=== Turn {L.turn} ===\n{L.grid_text}\nChats: {L.chats}\nActions: {L.actions}\nInv: {L.inventories}\nCrafted: {L.crafted}")
    print("\n== Summary ==")
    print("mode:", cfg.mode.value)
    print("success:", any(L.crafted.get(ITEM_BREAD,0)>=1 for L in res.logs))
    print("turns:", res.logs[-1].turn if res.logs else 0)

if __name__ == '__main__':
    main()
