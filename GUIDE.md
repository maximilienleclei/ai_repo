# AI Research Codebase Guide

This guide provides comprehensive documentation for understanding and working with this ML research repository.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration System](#configuration-system)
4. [Running Projects](#running-projects)
5. [Key Files Reference](#key-files-reference)
6. [Adding New Projects](#adding-new-projects)

---

## Overview

### Project Purpose
This is an ML research repository built on **Hydra-zen** configuration framework and **PyTorch Lightning**, designed for rapid experimentation across multiple research areas.

### Directory Structure
```
ai_research/
├── common/                # Shared utilities and base classes
│   ├── config.py          # Base Hydra configuration classes
│   ├── runner.py          # BaseTaskRunner
│   ├── store.py           # Config store utilities
│   ├── dl/                # Deep Learning utilities
│   │   ├── runner.py      # DeepLearningTaskRunner
│   │   ├── train.py       # Main training loop
│   │   ├── datamodule/    # Base data modules
│   │   ├── litmodule/     # Base Lightning modules & neural networks
│   │   └── utils/         # Lightning utilities
│   ├── ne/                # Neuroevolution utilities
│   └── utils/             # General utilities (wandb, beartype, etc.)
├── projects/              # Individual research projects
│   ├── classify_mnist/    # MNIST classification (functional)
│   ├── haptic_pred/       # Haptic prediction (gitignored)
│   └── ne_control_score/  # Neuroevolution control
├── data/                  # Data and outputs (gitignored)
├── Dockerfile             # Multi-mode container (cpu/cuda/7800xt)
├── requirements.txt       # Python dependencies
└── GUIDE.md              # This file
```

### Key Technologies
- **Hydra + Hydra-zen**: Configuration management with composable configs
- **PyTorch Lightning**: Training loop abstraction, checkpointing, logging
- **Beartype + jaxtyping**: Runtime type checking and tensor annotations
- **Weights & Biases**: Experiment tracking and visualization
- **Docker**: Multi-GPU backend support (CPU, CUDA, ROCm)

---

## Architecture

### Class Hierarchy

The codebase follows a three-layer inheritance pattern:

```
Layer 1: Base Classes (common/)
  BaseTaskRunner          # Abstract task runner
  BaseSubtaskConfig       # Base configuration dataclass

Layer 2: Domain-Specific (common/dl/ and common/ne/)
  DeepLearningTaskRunner  # Extends BaseTaskRunner for DL
  DeepLearningTaskConfig  # DL-specific config

Layer 3: Project-Specific (projects/*/)
  TaskRunner              # Inherits from DeepLearningTaskRunner
```

### Execution Flow

```
TaskRunner.run_task()
    ↓
handle_configs()              # Register all configs to Hydra store
    ↓
zen(run_subtask).hydra_main() # Launch Hydra with config
    ↓
run_subtask()                 # Execute training/evaluation
    ↓
train()                       # Main training loop (common/dl/train.py)
```

### Lightning Module Architecture

```
LightningModule (PyTorch Lightning)
    ↓
BaseLitModule (common/dl/litmodule/base.py)
    ├── Separates nn.Module (nnmodule) from Lightning logic
    ├── Manages optimizer and scheduler
    ├── Handles W&B logging with incremental tables
    └── Custom checkpoint state (curr_train_step, curr_val_epoch)
    ↓
BaseClassificationLitModule (common/dl/litmodule/classification.py)
    ├── Implements step() for classification
    ├── Cross-entropy loss
    ├── MulticlassAccuracy tracking
    └── Sample predictions to W&B
    ↓
MNISTClassificationLitModule (projects/classify_mnist/litmodule.py)
    └── Custom image unnormalization for W&B visualization
```

### Neural Network Modules

Neural network architectures live in `common/dl/litmodule/_nnmodule/` and are **separate** from Lightning modules:

- `feedforward.py` - FNN/MLP architectures
- `mamba2.py` - Mamba2 State Space Models
- `cond_autoreg/` - Conditional autoregressive models
- `cond_diffusion/` - Conditional diffusion models (DiT)

Example FNN from `common/dl/litmodule/_nnmodule/feedforward.py:17`:
```python
class FNN(nn.Module):
    def __init__(self, config: FNNConfig):
        # input_size -> [hidden_size x (num_layers-1)] -> output_size
        # ReLU activation between layers (not after output)
```

---

## Configuration System

### How Hydra-Zen Works

Configurations are managed through a **two-level system**:

1. **Python Config Stores** - Registered via `store_configs()` methods
2. **YAML Task Files** - Compose and override configs

### Task Parameter Resolution

When you run `python projects/classify_mnist/train.py task=fnn`:

1. **Parse task name** (`common/utils/runner.py:18`) - Extracts `fnn` from `task=fnn`
2. **Locate YAML file** - Finds `projects/classify_mnist/task/fnn.yaml`
3. **Compose configuration** - Hydra merges defaults and overrides
4. **Instantiate components** - Creates datamodule, litmodule, trainer, etc.

### Configuration Groups

| Group | Purpose | Examples |
|-------|---------|----------|
| `hydra/launcher` | Execution backend | `local`, `slurm` |
| `trainer` | PyTorch Lightning Trainer | `base` |
| `datamodule` | Data loading | `classify_mnist` |
| `litmodule` | Lightning module | `classify_mnist` |
| `litmodule/nnmodule` | Neural network | `fnn`, `mlp`, `mamba2` |
| `litmodule/optimizer` | Optimizer | `adamw`, `adam`, `sgd` |
| `litmodule/scheduler` | LR scheduler | `constant`, `linear_warmup` |
| `logger` | Experiment logging | `wandb` |

### Example YAML Configuration

File: `projects/classify_mnist/task/fnn.yaml`
```yaml
# @package _global_
defaults:
  - /datamodule: classify_mnist       # Use registered datamodule
  - /litmodule: classify_mnist        # Use registered litmodule
  - override /hydra/launcher: local   # Run locally (not SLURM)
  - _self_                            # This file's overrides apply last

hydra:
  launcher:
    cpus_per_task: 4

config:
  device: cpu                         # cpu or gpu

litmodule:
  config:
    num_classes: 10                   # 10 digit classes
  nnmodule:                           # Neural network config
    config:
      input_size: 784                 # 28x28 flattened
      output_size: 10
      num_layers: 1                   # Single linear layer
  optimizer:
    lr: 0.002

logger:
  entity: maximilienlc                # W&B entity

trainer:
  max_epochs: 3
```

### Config Store Registration

Projects register configs in `store_configs()` method:

```python
# From projects/classify_mnist/train.py:17
class TaskRunner(DeepLearningTaskRunner):
    @classmethod
    def store_configs(cls, store: ZenStore):
        super().store_configs(store=store)

        # Register datamodule
        store(
            generate_config(
                MNISTDataModule,
                config=generate_config(MNISTDataModuleConfig),
                dataloader=generate_config_partial(DataLoader),
            ),
            name="classify_mnist",
            group="datamodule",
        )

        # Register litmodule
        store(
            generate_config(
                MNISTClassificationLitModule,
                config=generate_config(BaseClassificationLitModuleConfig),
            ),
            name="classify_mnist",
            group="litmodule",
        )
```

### Environment Variables

- **`AI_RESEARCH_PATH`**: Root path for data storage and outputs
  - Used in: `common/config.py:28`, `common/utils/wandb.py:11`
  - Default data dir: `${AI_RESEARCH_PATH}/data/`
  - Hydra outputs: `${AI_RESEARCH_PATH}/data/${project}/${task}/`

- **`PYTHONPATH`**: Should include `${AI_RESEARCH_PATH}` for imports

---

## Running Projects

### Container Modes

The Dockerfile supports three modes:

```bash
# Build for CPU
docker build --build-arg MODE=cpu -t ai-research:cpu_latest .

# Build for NVIDIA CUDA
docker build --build-arg MODE=cuda -t ai-research:cuda_latest .

# Build for AMD Radeon RX 7800 XT
docker build --build-arg MODE=7800xt -t ai-research:7800xt_latest .
```

### Running classify_mnist

#### With Podman (tested command):
```bash
podman run --rm \
  -e AI_RESEARCH_PATH=${AI_RESEARCH_PATH} \
  -e PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH} \
  -v ${AI_RESEARCH_PATH}:${AI_RESEARCH_PATH} \
  -v /dev/shm:/dev/shm \
  -w ${AI_RESEARCH_PATH} \
  ai-research:cpu_latest \
  python projects/classify_mnist/train.py task=fnn
```

#### With Docker + GPU:
```bash
docker run --gpus all --rm \
  -e AI_RESEARCH_PATH=${AI_RESEARCH_PATH} \
  -e PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH} \
  -v ${AI_RESEARCH_PATH}:${AI_RESEARCH_PATH} \
  -v /dev/shm:/dev/shm \
  -w ${AI_RESEARCH_PATH} \
  ai-research:cuda_latest \
  python projects/classify_mnist/train.py task=mlp
```

### Available Tasks

For `classify_mnist`:
- `task=fnn` - Single linear layer (logistic regression)
- `task=mlp` - Multi-layer perceptron
- `task=mlp_beluga` - SLURM configuration for Compute Canada

### Output Locations

Hydra outputs are saved to:
```
${AI_RESEARCH_PATH}/data/${project}/${task}/
└── <timestamp>/
    ├── .hydra/              # Hydra config snapshots
    ├── lightning/           # PyTorch Lightning checkpoints
    └── <task>.log          # Training logs
```

### Weights & Biases

W&B logging is automatic:
- API key location: `${AI_RESEARCH_PATH}/WANDB_KEY.txt`
- Login handled in: `common/utils/wandb.py:7`
- Project naming: `${project}_${task}`
- Supports offline nodes via `wandb-osh`

---

## Key Files Reference

### Core Training Loop
- **`common/dl/train.py:18`** - `train()` function
  - Seeds random generators
  - Instantiates Trainer, datamodule, litmodule
  - Auto-tunes batch size and num_workers
  - Optionally compiles model with `torch.compile`
  - Calls `trainer.fit()` and `trainer.validate()`

### Base Configuration
- **`common/config.py:18`** - `BaseSubtaskConfig`
  - Common fields: `seed`, `device`, `data_dir`, `output_dir`, `compile`
- **`common/config.py:63`** - `BaseHydraConfig`
  - Hydra sweep directory configuration

### Base Runner
- **`common/runner.py:15`** - `BaseTaskRunner`
  - `run_task()` - Main entry point
  - `handle_configs()` - Register configs to Hydra store
  - `run_subtask()` - Execute training/evaluation

### Deep Learning Runner
- **`common/dl/runner.py:14`** - `DeepLearningTaskRunner`
  - Extends BaseTaskRunner
  - Overrides `store_configs()` to register DL components
  - `run_subtask()` calls `train()` function

### Base Lightning Module
- **`common/dl/litmodule/base.py:47`** - `BaseLitModule`
  - Separates `nn.Module` (nnmodule) from Lightning logic
  - `configure_optimizers()` - Returns optimizer and scheduler
  - `training_step()`, `validation_step()`, `test_step()`
  - `on_save_checkpoint()`, `on_load_checkpoint()` - Custom state
  - W&B incremental table logging

### Classification Module
- **`common/dl/litmodule/classification.py:21`** - `BaseClassificationLitModule`
  - Implements `step()` for classification
  - Cross-entropy loss computation
  - MulticlassAccuracy tracking
  - Sample prediction logging to W&B

### Base Data Module
- **`common/dl/datamodule/base.py:27`** - `BaseDataModule`
  - Wraps PyTorch Lightning's `LightningDataModule`
  - `Datasets` dataclass for train/val/test splits
  - Abstract methods: `prepare_data()`, `setup()`

### Utilities
- **`common/utils/runner.py:18`** - `get_task_name()`, `get_project_name()`, `get_absolute_project_path()`
- **`common/utils/wandb.py:7`** - `login_wandb()`
- **`common/dl/utils/lightning.py:13`** - `instantiate_trainer()`, `set_batch_size_and_num_workers()`
- **`common/utils/hydra_zen.py:12`** - `generate_config()`, `generate_config_partial()`, `destructure()`

### Neural Network Modules
- **`common/dl/litmodule/_nnmodule/feedforward.py:17`** - `FNN` (Feedforward Neural Network)
- **`common/dl/litmodule/_nnmodule/mamba2.py:22`** - `Mamba2` (State Space Model)
- **`common/dl/litmodule/_nnmodule/store.py:10`** - Config registration for NN modules

---

## Adding New Projects

### Required Files

To create a new project at `projects/my_project/`:

```
projects/my_project/
├── train.py              # Entry point with TaskRunner
├── litmodule.py          # Lightning module (model definition)
├── datamodule.py         # Data loading
└── task/
    └── my_task.yaml      # Hydra configuration
```

### Step-by-Step Guide

#### 1. Create `train.py`

```python
from common.dl.runner import DeepLearningTaskRunner
from hydra_zen import ZenStore, generate_config, generate_config_partial
from torch.utils.data import DataLoader

from .litmodule import MyLitModule, MyLitModuleConfig
from .datamodule import MyDataModule, MyDataModuleConfig


class TaskRunner(DeepLearningTaskRunner):

    @classmethod
    def store_configs(cls, store: ZenStore):
        super().store_configs(store=store)

        # Register datamodule
        store(
            generate_config(
                MyDataModule,
                config=generate_config(MyDataModuleConfig),
                dataloader=generate_config_partial(DataLoader),
            ),
            name="my_project",
            group="datamodule",
        )

        # Register litmodule
        store(
            generate_config(
                MyLitModule,
                config=generate_config(MyLitModuleConfig),
            ),
            name="my_project",
            group="litmodule",
        )


if __name__ == "__main__":
    TaskRunner.run_task()
```

#### 2. Create `litmodule.py`

```python
from dataclasses import dataclass
from common.dl.litmodule.base import BaseLitModule, BaseLitModuleConfig
import torch
import torch.nn as nn


@dataclass
class MyLitModuleConfig(BaseLitModuleConfig):
    # Add project-specific config fields
    pass


class MyLitModule(BaseLitModule):

    def __init__(self, config: MyLitModuleConfig):
        super().__init__(config=config)

    def step(self, batch, batch_idx):
        """Implement forward pass and loss computation."""
        x, y = batch
        y_hat = self.nnmodule(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return {"loss": loss}
```

#### 3. Create `datamodule.py`

```python
from dataclasses import dataclass
from common.dl.datamodule.base import BaseDataModule, BaseDataModuleConfig


@dataclass
class MyDataModuleConfig(BaseDataModuleConfig):
    # Add project-specific config fields
    pass


class MyDataModule(BaseDataModule):

    def __init__(self, config, dataloader):
        super().__init__(config=config, dataloader=dataloader)

    def prepare_data(self):
        """Download or prepare data (runs once per node)."""
        pass

    def setup(self, stage):
        """Create datasets for train/val/test."""
        if stage == "fit":
            self.datasets.train = ...  # Your train dataset
            self.datasets.val = ...    # Your val dataset
```

#### 4. Create `task/my_task.yaml`

```yaml
# @package _global_
defaults:
  - /datamodule: my_project
  - /litmodule: my_project
  - override /hydra/launcher: local
  - _self_

hydra:
  launcher:
    cpus_per_task: 4

config:
  device: cpu

litmodule:
  nnmodule:
    config:
      input_size: 100
      output_size: 10
  optimizer:
    lr: 0.001

trainer:
  max_epochs: 10
```

#### 5. Run Your Project

```bash
python projects/my_project/train.py task=my_task
```

### Tips

1. **Reuse Base Classes**: Inherit from `BaseClassificationLitModule` or `BaseLitModule` depending on your task
2. **Leverage Existing NNModules**: Use FNN, MLP, Mamba2 from `common/dl/litmodule/_nnmodule/`
3. **Follow Naming Conventions**: Use `name="<project_name>"` when registering configs
4. **Test Data Loading**: Create a `datamodule_test.py` with pytest (see `projects/classify_mnist/datamodule_test.py`)
5. **Check Git Status**: Current modified files are `projects/classify_mnist/litmodule.py` and `requirements.txt`

---

## Additional Notes

### Testing
- Framework: pytest
- Example: `projects/classify_mnist/datamodule_test.py`
- No CI/CD configured

### SLURM Integration
- Uses custom fork: `git+https://github.com/maximilienleclei/hydra`
- Example HPC config: `projects/classify_mnist/task/mlp_beluga.yaml`
- Switch via: `override /hydra/launcher: slurm`

### Type Safety
- **Beartype**: Runtime type checking (`@beartype_this_package` in `__init__.py:8`)
- **Jaxtyping**: Tensor shape/dtype annotations
- Custom validators in `common/utils/beartype.py`: `ge()`, `not_empty()`, `one_of()`

### Container Features
- **CPU mode**: Default, lightweight
- **CUDA mode**: NVIDIA CUDA 12.8 toolkit
- **7800xt mode**: AMD ROCm 6.4.2 with gfx1101 architecture tricks
- Uses `uv` for fast dependency installation

---

## Troubleshooting

### Common Issues

1. **"You must specify the task argument"**
   - Ensure you pass `task=<name>` when running
   - Check that `task/<name>.yaml` exists

2. **ModuleNotFoundError**
   - Set `PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH}`
   - Ensure `AI_RESEARCH_PATH` points to repo root

3. **W&B login failure**
   - Check `${AI_RESEARCH_PATH}/WANDB_KEY.txt` exists
   - File should contain your W&B API key

4. **GPU not detected (AMD)**
   - For 7800XT, use `MODE=7800xt` build
   - Check `libhsa-runtime64.so` was deleted from torch libs

5. **SLURM job fails**
   - Verify data is synced to HPC cluster
   - Check compute node has sufficient resources

---

## Quick Reference

### Run Training
```bash
python projects/<project>/train.py task=<task_name>
```

### Key Environment Variables
```bash
export AI_RESEARCH_PATH=/path/to/repo
export PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH}
```

### Config Override Examples
```bash
# Change learning rate
python projects/classify_mnist/train.py task=fnn litmodule.optimizer.lr=0.01

# Run on GPU
python projects/classify_mnist/train.py task=fnn config.device=gpu

# Change max epochs
python projects/classify_mnist/train.py task=fnn trainer.max_epochs=10
```

### View Hydra Config
```bash
python projects/classify_mnist/train.py task=fnn --help
python projects/classify_mnist/train.py task=fnn --cfg job
```

---

**Last Updated**: 2025-12-16 (from git commit 2ac8046)
**Repository**: https://github.com/maximilienleclei/ai_research (inferred)
