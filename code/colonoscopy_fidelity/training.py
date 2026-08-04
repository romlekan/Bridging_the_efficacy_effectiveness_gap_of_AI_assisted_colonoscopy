import logging
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from colonoscopy_fidelity.network import MultiOutcomeModel, NetworkOutput, gradient_norm, parameter_groups, total_training_loss


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 2025
    epochs: int = 20000
    batch_size: int = 8192
    gradient_accumulation: int = 64
    learning_rate: float = 1e-7
    weight_decay: float = 1e-5
    warmup_steps: int = 100000
    total_steps: int = 10000000
    minimum_learning_rate_ratio: float = 0.001
    gradient_clip: float = 0.1
    log_interval: int = 1000
    save_interval: int = 100000


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    optimizer_step: int = 0
    samples_seen: int = 0
    best_loss: float = math.inf
    running_loss: float = 0.0
    running_batches: int = 0
    started_at: float = field(default_factory=time.time)

    def update(self, loss: float, batch_size: int) -> None:
        self.running_loss += loss
        self.running_batches += 1
        self.samples_seen += batch_size
        self.global_step += 1

    def mean_loss(self) -> float:
        if self.running_batches == 0:
            return math.nan
        return self.running_loss / self.running_batches

    def reset_window(self) -> None:
        self.running_loss = 0.0
        self.running_batches = 0


@dataclass(frozen=True)
class CohortBatch:
    continuous: Tensor
    observed: Tensor
    categorical: Optional[Tensor]
    binary_target: Tensor
    continuous_target: Tensor
    fidelity_target: Tensor
    binary_mask: Tensor
    continuous_mask: Tensor
    fidelity_observed: Tensor

    def to(self, device: torch.device) -> "CohortBatch":
        return CohortBatch(
            continuous=self.continuous.to(device, non_blocking=True),
            observed=self.observed.to(device, non_blocking=True),
            categorical=self.categorical.to(device, non_blocking=True) if self.categorical is not None else None,
            binary_target=self.binary_target.to(device, non_blocking=True),
            continuous_target=self.continuous_target.to(device, non_blocking=True),
            fidelity_target=self.fidelity_target.to(device, non_blocking=True),
            binary_mask=self.binary_mask.to(device, non_blocking=True),
            continuous_mask=self.continuous_mask.to(device, non_blocking=True),
            fidelity_observed=self.fidelity_observed.to(device, non_blocking=True),
        )

    @property
    def size(self) -> int:
        return self.continuous.shape[0]


class CohortTensorDataset(Dataset[Mapping[str, Tensor]]):
    def __init__(self, tensors: Mapping[str, Tensor]) -> None:
        if not tensors:
            raise ValueError("tensors cannot be empty")
        lengths = {tensor.shape[0] for tensor in tensors.values()}
        if len(lengths) != 1:
            raise ValueError("tensor lengths differ")
        self.tensors = dict(tensors)
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Mapping[str, Tensor]:
        return {name: tensor[index] for name, tensor in self.tensors.items()}


def collate(records: Sequence[Mapping[str, Tensor]]) -> CohortBatch:
    def stack(name: str) -> Tensor:
        return torch.stack([record[name] for record in records], dim=0)

    categorical = stack("categorical") if "categorical" in records[0] else None
    return CohortBatch(
        continuous=stack("continuous"),
        observed=stack("observed"),
        categorical=categorical,
        binary_target=stack("binary_target"),
        continuous_target=stack("continuous_target"),
        fidelity_target=stack("fidelity_target"),
        binary_mask=stack("binary_mask"),
        continuous_mask=stack("continuous_mask"),
        fidelity_observed=stack("fidelity_observed"),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if distributed_available() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if distributed_available() else 1


def primary() -> bool:
    return rank() == 0


def initialize_distributed() -> Tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    process_rank = int(os.environ.get("RANK", "0"))
    process_world = int(os.environ.get("WORLD_SIZE", "1"))
    if process_world > 1 and not distributed_available():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return process_rank, process_world, device


def synchronize() -> None:
    if distributed_available():
        torch.distributed.barrier()


def reduce_mean(value: Tensor) -> Tensor:
    if not distributed_available():
        return value
    result = value.detach().clone()
    torch.distributed.all_reduce(result, op=ReduceOp.SUM)
    return result / world_size()


def cosine_schedule(step: int, settings: TrainingSettings) -> float:
    if step < settings.warmup_steps:
        return max(1e-12, step / max(1, settings.warmup_steps))
    progress = (step - settings.warmup_steps) / max(1, settings.total_steps - settings.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return settings.minimum_learning_rate_ratio + (1.0 - settings.minimum_learning_rate_ratio) * cosine


def optimizer_for(model: nn.Module, settings: TrainingSettings) -> AdamW:
    return AdamW(
        parameter_groups(model, settings.weight_decay),
        lr=settings.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def scheduler_for(optimizer: Optimizer, settings: TrainingSettings) -> LambdaLR:
    return LambdaLR(optimizer, lambda step: cosine_schedule(step, settings))


def loader_for(dataset: Dataset[Mapping[str, Tensor]], settings: TrainingSettings, workers: int = 16) -> DataLoader[CohortBatch]:
    sampler = DistributedSampler(dataset, shuffle=True, seed=settings.seed, drop_last=True) if distributed_available() else None
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        collate_fn=collate,
    )


def model_state(model: nn.Module) -> Mapping[str, Tensor]:
    target = model.module if isinstance(model, DistributedDataParallel) else model
    return target.state_dict()


def rng_state() -> Mapping[str, object]:
    value: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all(value["cuda"])


def atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_training_state(path: Path, model: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, state: TrainingState, settings: TrainingSettings) -> None:
    if not primary():
        return
    value: Mapping[str, object] = {
        "model": model_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": asdict(state),
        "settings": asdict(settings),
        "rng": rng_state(),
        "seed": settings.seed,
    }
    atomic_torch_save(path, value)


def load_training_state(path: Path, model: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, device: torch.device) -> TrainingState:
    value = torch.load(path, map_location=device)
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    restore_rng(value["rng"])
    return TrainingState(**value["training_state"])


class Trainer:
    def __init__(self, model: MultiOutcomeModel, settings: TrainingSettings, device: torch.device, output: Path) -> None:
        self.settings = settings
        self.device = device
        self.output = output
        model = model.to(device)
        self.model: nn.Module = DistributedDataParallel(model, device_ids=[device.index]) if distributed_available() and device.type == "cuda" else model
        self.optimizer = optimizer_for(self.model, settings)
        self.scheduler = scheduler_for(self.optimizer, settings)
        self.state = TrainingState()

    def forward(self, batch: CohortBatch) -> Mapping[str, Tensor]:
        output = self.model(batch.continuous, batch.observed, batch.categorical)
        if not isinstance(output, NetworkOutput):
            raise TypeError("model returned unexpected output")
        return total_training_loss(
            output,
            batch.binary_target,
            batch.continuous_target,
            batch.fidelity_target,
            batch.binary_mask,
            batch.continuous_mask,
            batch.fidelity_observed,
        )

    def train_batch(self, batch: CohortBatch) -> Mapping[str, float]:
        moved = batch.to(self.device)
        losses = self.forward(moved)
        scaled = losses["total"] / self.settings.gradient_accumulation
        scaled.backward()
        should_step = (self.state.global_step + 1) % self.settings.gradient_accumulation == 0
        norm = gradient_norm(self.model)
        if should_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state.optimizer_step += 1
        reduced = {name: float(reduce_mean(value.detach()).cpu()) for name, value in losses.items()}
        reduced["gradient_norm"] = float(reduce_mean(norm.detach()).cpu())
        reduced["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        self.state.update(reduced["total"], batch.size)
        return reduced

    def train_epoch(self, loader: Iterable[CohortBatch]) -> Mapping[str, float]:
        self.model.train()
        aggregates: MutableMapping[str, float] = {}
        batches = 0
        for batch in loader:
            metrics = self.train_batch(batch)
            batches += 1
            for name, value in metrics.items():
                aggregates[name] = aggregates.get(name, 0.0) + value
            if self.state.global_step % self.settings.log_interval == 0 and primary():
                LOGGER.info("step=%d loss=%.8f lr=%.12f", self.state.global_step, self.state.mean_loss(), metrics["learning_rate"])
                self.state.reset_window()
            if self.state.global_step % self.settings.save_interval == 0:
                save_training_state(self.output / "latest.pt", self.model, self.optimizer, self.scheduler, self.state, self.settings)
            if self.state.optimizer_step >= self.settings.total_steps:
                break
        self.state.epoch += 1
        return {name: total / max(1, batches) for name, total in aggregates.items()}

    def fit(self, loader: Iterable[CohortBatch]) -> TrainingState:
        set_seed(self.settings.seed + rank())
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(self.state.epoch, self.settings.epochs):
            metrics = self.train_epoch(loader)
            if primary():
                LOGGER.info("epoch=%d total=%.8f", self.state.epoch, metrics.get("total", math.nan))
            if self.state.optimizer_step >= self.settings.total_steps:
                break
        save_training_state(self.output / "final.pt", self.model, self.optimizer, self.scheduler, self.state, self.settings)
        synchronize()
        return self.state

    def resume(self, path: Path) -> None:
        self.state = load_training_state(path, self.model, self.optimizer, self.scheduler, self.device)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
