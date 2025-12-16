# ai_research

**Reduces code & configuration boilerplate with:**
* [Hydra](https://github.com/facebookresearch/hydra) for task configuration.
* [Hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) for
[Hydra](https://github.com/facebookresearch/hydra) structured configuration
management.
* [Lightning](https://github.com/Lightning-AI/pytorch-lightning) for
[PyTorch](https://github.com/pytorch/pytorch) code.

**Simplifies machine learning workflows:**
* SLURM job definition, queuing and monitoring with
[Submitit](https://github.com/facebookincubator/submitit) through its
[Hydra Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
* [Podman](https://podman.io/) environment containerization for both regular &
SLURM-based execution.
* Transition from regular execution to SLURM-based execution by swapping
as little as a single [Hydra](https://github.com/facebookresearch/hydra)
configuration field.

**Facilitates experimentation development through:**
* An object-oriented common/project/task structure.
* Deeply integrated logging with [Weights & Biases](https://wandb.ai/site).

**Promotes high-quality and reproducible code by:**
* Formatting with [Black](https://github.com/psf/black).
* DType & Shape type hinting for [PyTorch](https://github.com/pytorch/pytorch)
tensors using [jaxtyping](https://github.com/google/jaxtyping).
* Dynamic type-checking w/ [Beartype](https://github.com/beartype/beartype).