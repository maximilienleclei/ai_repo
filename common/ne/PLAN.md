# Pop Design Plan

## Goal

Design `BasePop` as a wrapper around nets:
1. Nets = pure low-level computation (no argmax, no action selection)
2. Generator-discriminator setup where all agents can do both
3. Dual memory for recurrent/dynamic nets (separate gen vs disc state)
4. Pop manages memory externally (nets receive state, don't store it)
5. Pop maintains inheritable state (env, mem, fitness) for GA

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           EVAL                                   │
│  Runs episodes, computes fitness                                 │
│  For adversarial: random matching, 2×N fitness per agent         │
└───────────────────────────────┬─────────────────────────────────┘
                                │ fitness_scores
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                            ALG                                   │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │        GA           │    │             ES                  │ │
│  │                     │    │                                 │ │
│  │ Input: fitness      │    │ Input: params [pop, num_params] │ │
│  │        (+ inherited)│    │        fitness [pop]            │ │
│  │                     │    │                                 │ │
│  │ Output: indices     │    │ Output: new_params              │ │
│  │   e.g. [1,1,3,3]    │    │   [pop, num_params]             │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                            POP                                   │
│                                                                  │
│  ┌──────────────────┐   ┌────────────────────────────────────┐  │
│  │   GeneratorPop   │   │   GeneratorDiscriminatorPop        │  │
│  │                  │   │                                    │  │
│  │ • generate(x)    │   │ • generate(x)   [uses gen_mem]     │  │
│  │ • single memory  │   │ • discriminate(x) [uses disc_mem]  │  │
│  └──────────────────┘   └────────────────────────────────────┘  │
│                                                                  │
│  For GA: maintains inheritable state (env, mem, fitness)         │
│  For ES: exposes/reconstructs from flattened parameters          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           NETS                                   │
│  Pure computation: (x, mem) → (out, mem)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice |
|----------|--------|
| Gen-only vs Gen+Disc | Separate classes: `GeneratorPop`, `GeneratorDiscriminatorPop` |
| Memory management | Pop manages externally, passes mem to nets |
| Inheritance (GA only) | Config-driven toggles: `inherit_env`, `inherit_mem`, `inherit_fit` |
| GA alg output | List of indices `[1,1,3,3]` - Pop handles copying/mutation |
| ES alg output | New parameters matrix - Pop reconstructs nets |
| Adversarial fitness | 2×N scores per agent (N as gen, N as disc), summed |

---

## Core Concepts

### Pop Class Hierarchy

```
BasePop (abstract)
│   • nets: BaseNets
│   • mutate()
│   • For GA: apply_selection(indices), get_inheritable_fitness()
│   • For ES: get_params(), set_params()
│
├── GeneratorPop
│   • generate(x) → raw outputs
│   • _mem: single memory slot
│   • For GA: inheritable mem state
│
└── GeneratorDiscriminatorPop
    • generate(x) → first n outputs [uses gen_mem]
    • discriminate(x) → last m outputs [uses disc_mem]
    • _gen_mem, _disc_mem: dual memory slots
    • For GA: inheritable gen_mem and disc_mem
```

### Output Partitioning (GeneratorDiscriminatorPop only)

Network output shape: `[num_nets, batch_size, num_outputs]`
- `output[..., :n_gen]` = generation outputs (actions, logits, etc.)
- `output[..., -n_disc:]` = discrimination outputs (real/fake score)

### Memory

**GeneratorPop**: Single `_mem` slot

**GeneratorDiscriminatorPop**: Dual slots
- `_gen_mem` - state used during generation
- `_disc_mem` - state used during discrimination

Memory types by net:
- **Feedforward**: None (stateless)
- **Recurrent static**: list of hidden tensors per layer
- **Dynamic**: `n_mean_m2_x_z` tensor (per-node z-scores)

### Inheritance (GA only)

Three toggles, each independent:

| Toggle | What inherits | Stored in |
|--------|---------------|-----------|
| `inherit_env` | Environment state | Pop (or Eval?) |
| `inherit_mem` | Memory state after parent finished | Pop |
| `inherit_fit` | Accumulated fitness | Pop |

**Flow:**
1. Eval computes `fitness_scores` for current generation
2. If `inherit_fit`: `total_fitness = fitness_scores + inherited_fitness`
3. GA alg receives `total_fitness`, returns `indices` (e.g., `[1,1,3,3]`)
4. Pop receives `indices`, performs:
   - Copy net weights from parent to child
   - If `inherit_mem`: copy parent's final mem to child
   - If `inherit_fit`: copy parent's accumulated fitness to child
   - If `inherit_env`: (handled by Eval - saves/restores env state)
5. Mutation applied to children

---

## Pseudocode: Pop Classes

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor

@dataclass
class GAConfig:
    """Config for GA inheritance (optional, only for GA algs)."""
    inherit_env: bool = False
    inherit_mem: bool = False
    inherit_fit: bool = False


class BasePop(ABC):
    """
    Abstract base for all Pop classes.
    Wraps nets, delegates mutation, provides GA/ES interfaces.
    """

    def __init__(self, nets: BaseNets, ga_config: GAConfig | None = None):
        self.nets = nets
        self.ga_config = ga_config

        # For GA inheritance
        self._inherited_fitness: Tensor | None = None  # [num_nets]

    # ── Abstract ──────────────────────────────────────────────

    @abstractmethod
    def reset_memory(self) -> None: ...

    @abstractmethod
    def save_memory(self) -> Any: ...

    @abstractmethod
    def restore_memory(self, state: Any) -> None: ...

    # ── Delegation ────────────────────────────────────────────

    def mutate(self) -> None:
        self.nets.mutate()

    @property
    def num_nets(self) -> int:
        return self.nets.config.num_nets

    # ── GA interface ──────────────────────────────────────────

    def apply_selection(self, indices: list[int]) -> None:
        """
        Apply GA selection. indices[i] = which parent agent i inherits from.
        E.g., [1,1,3,3] means agent 0 copies from 1, agent 2 copies from 3.
        """
        # Copy net weights
        self.nets.apply_selection(indices)

        # Copy inheritable state based on config
        if self.ga_config and self.ga_config.inherit_mem:
            self._inherit_memory(indices)
        if self.ga_config and self.ga_config.inherit_fit:
            self._inherit_fitness(indices)

    def get_total_fitness(self, eval_fitness: Tensor) -> Tensor:
        """Combine eval fitness with inherited fitness if enabled."""
        if self.ga_config and self.ga_config.inherit_fit and self._inherited_fitness is not None:
            return eval_fitness + self._inherited_fitness
        return eval_fitness

    # ── ES interface ──────────────────────────────────────────

    def get_params(self) -> Tensor:
        """Return flattened parameters [num_nets, num_params] for ES."""
        return self.nets.get_params()

    def set_params(self, params: Tensor) -> None:
        """Reconstruct nets from flattened parameters."""
        self.nets.set_params(params)


class GeneratorPop(BasePop):
    """Pop for generate-only tasks (e.g., action classification)."""

    def __init__(self, nets: BaseNets, ga_config: GAConfig | None = None):
        super().__init__(nets, ga_config)
        self._mem: Any = None

    def generate(self, x: Tensor) -> Tensor:
        """Forward pass. Returns all outputs."""
        out, self._mem = self.nets(x, self._mem)
        return out

    def reset_memory(self) -> None:
        self._mem = None

    def save_memory(self) -> Any:
        return _clone_state(self._mem)

    def restore_memory(self, state: Any) -> None:
        self._mem = _clone_state(state)

    def _inherit_memory(self, indices: list[int]) -> None:
        # Copy parent mem to child based on indices
        ...


class GeneratorDiscriminatorPop(BasePop):
    """Pop for adversarial tasks (all agents generate AND discriminate)."""

    def __init__(self, nets: BaseNets, n_gen: int, n_disc: int, ga_config: GAConfig | None = None):
        super().__init__(nets, ga_config)
        self.n_gen = n_gen
        self.n_disc = n_disc
        self._gen_mem: Any = None
        self._disc_mem: Any = None

    def generate(self, x: Tensor) -> Tensor:
        """Forward pass using gen memory. Returns first n_gen outputs."""
        out, self._gen_mem = self.nets(x, self._gen_mem)
        return out[..., :self.n_gen]

    def discriminate(self, x: Tensor) -> Tensor:
        """Forward pass using disc memory. Returns last n_disc outputs."""
        out, self._disc_mem = self.nets(x, self._disc_mem)
        return out[..., -self.n_disc:]

    def reset_memory(self, mode: str | None = None) -> None:
        """Reset memory. mode='gen', 'disc', or None (both)."""
        if mode is None or mode == 'gen':
            self._gen_mem = None
        if mode is None or mode == 'disc':
            self._disc_mem = None

    def save_memory(self) -> dict[str, Any]:
        return {'gen': _clone_state(self._gen_mem), 'disc': _clone_state(self._disc_mem)}

    def restore_memory(self, state: dict[str, Any]) -> None:
        self._gen_mem = _clone_state(state['gen'])
        self._disc_mem = _clone_state(state['disc'])

    def _inherit_memory(self, indices: list[int]) -> None:
        # Copy parent gen_mem and disc_mem to child based on indices
        ...


def _clone_state(state: Any) -> Any:
    """Clone state (handles None, Tensor, list of Tensors)."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.clone()
    if isinstance(state, list):
        return [s.clone() for s in state]
    raise TypeError(f"Unknown state type: {type(state)}")
```

---

## Pseudocode: BaseNets `__call__` changes

```python
class BaseNets(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, mem: Any = None) -> tuple[Tensor, Any]:
        """Forward pass. Returns (output, new_mem)."""
        ...

    def mutate(self) -> None: ...
```

For **FeedforwardStaticNets** - mem is ignored:
```python
def __call__(self, x: Tensor, mem: Any = None) -> tuple[Tensor, None]:
    # ... existing computation ...
    return output, None  # No memory
```

For **RecurrentStaticNets** - mem = list of hidden tensors per layer:
```python
def __call__(self, x: Tensor, mem: list[Tensor] | None = None) -> tuple[Tensor, list[Tensor]]:
    hidden = mem if mem is not None else self._init_hidden(x)
    out, new_hidden = self._forward_with_hidden(x, hidden)
    return out, new_hidden
```

For **DynamicNets** - mem = the `n_mean_m2_x_z` tensor:
```python
def __call__(self, x: Tensor, mem: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if mem is not None:
        self.wrs.n_mean_m2_x_z = mem
    # ... existing computation ...
    return output, self.wrs.n_mean_m2_x_z.clone()
```

---

## Open Design Questions (for later)

1. **Output processing**: How to add argmax, softmax, tanh on top of generate()? Separate classes? Mixins? Decorators?

2. **Episode boundaries**: Should Pop have `on_done(done_mask)` for selective reset? Or leave to eval code?

3. **inherit_env**: Where does env state live? Pop or Eval? Likely Eval since it owns the env.

---

## Files to Modify

| File | Changes |
|------|---------|
| `common/ne/pop/_nets/base.py` | Change `__call__` signature to `(x, mem) → (out, mem)` |
| `common/ne/pop/_nets/static/feedforward.py` | Update `__call__` to return `(out, None)` |
| `common/ne/pop/_nets/static/recurrent.py` | Update `__call__` with proper mem handling |
| `common/ne/pop/_nets/dynamic/base.py` | Update `__call__` with mem = `n_mean_m2_x_z` |
| `common/ne/pop/base.py` | Implement `BasePop`, `GeneratorPop`, `GeneratorDiscriminatorPop` |
| `common/ne/pop/config.py` (new?) | `GAConfig` dataclass |

---

## Incremental Implementation Steps

### Step 1: Change `__call__` signature in `BaseNets`

**What**: Change the signature from `(x) -> out` to `(x, mem=None) -> (out, mem)`.

```python
# Old
def __call__(self, x: Tensor) -> Tensor:

# New
def __call__(self, x: Tensor, mem: Any = None) -> tuple[Tensor, Any]:
```

**Breaking change**: All call sites need updating from `out = nets(x)` to `out, _ = nets(x)`.

**Why this is a good decision**:

1. **Clean API**: `nets(x, mem)` reads naturally. No separate method name to remember.

2. **Establishes the contract**: Pop calls `nets(x, mem)`, gets back `(out, new_mem)`. Consistent across all net types.

3. **Default does the right thing**: For feedforward nets, mem in = None, mem out = None.

4. **Forces awareness**: Call sites must acknowledge the return tuple.

---

### Step 2: Implement Pop classes

**What**: `BasePop`, `GeneratorPop`, `GeneratorDiscriminatorPop`

**Why separate classes** (not a single class with config flag):

1. **Clear intent**: `GeneratorPop` = generate-only tasks. `GeneratorDiscriminatorPop` = adversarial. No ambiguity.

2. **Different state shapes**: Generator has one mem slot. GenDisc has two. Simpler to model explicitly.

3. **No dead code**: GeneratorPop doesn't carry unused discriminate() method.

**Why `BasePop` provides GA/ES interfaces**:

1. **GA alg outputs indices** → Pop's `apply_selection(indices)` handles:
   - Copying net weights
   - Inheriting mem (if enabled)
   - Inheriting fitness (if enabled)

2. **ES alg needs params** → Pop's `get_params()` / `set_params()` handles:
   - Flattening/unflattening net weights

This keeps alg code simple: alg just does selection math, Pop handles the mechanics.

---

### Step 3 (later): Proper mem handling in recurrent/dynamic nets

**What**: Make recurrent nets actually use the mem parameter correctly.

**Why defer**:
- Feedforward nets work immediately (mem = None)
- Validate Pop design first with feedforward
- Recurrent fix is entangled with existing hidden state bug

---

### Step 4 (later): Inheritance implementation

**What**: Flesh out `_inherit_memory()`, `_inherit_fitness()`, and coordinate with Eval for env inheritance.

**Why defer**:
- Core forward pass more important first
- Inheritance only matters for GA
- Need to decide where env state lives (probably Eval)

---

## What We're NOT Doing Yet

- Recurrent/dynamic mem handling (Step 3)
- GA inheritance logic (Step 4)
- Output processing (argmax, tanh, etc.) - TBD
- Episode boundary handling - TBD
- Batch dimension unification - separate task
