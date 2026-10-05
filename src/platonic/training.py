"""Training and stable checkpoint serialization."""
from __future__ import annotations

import csv
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataConfig, Rules, build_rules, sample_corpus
from .model import CausalTransformer, TransformerConfig


class WSDLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup, stable plateau, and final cosine decay."""

    def __init__(self, optimizer, warmup_steps, decay_steps, total_steps, min_lr_factor):
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.total_steps = int(total_steps)
        self.min_lr_factor = float(min_lr_factor)
        self.decay_start = max(0, self.total_steps - self.decay_steps)
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.warmup_steps > 0 and step < self.warmup_steps:
            factor = step / self.warmup_steps
        elif self.decay_steps > 0 and step >= self.decay_start:
            progress = min(step - self.decay_start, self.decay_steps) / self.decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine
        else:
            factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]


def _log2_checkpoints(end: int, frequency: int) -> list[int]:
    current = 1.0
    factor = 2 ** (1.0 / frequency)
    threshold = 2 ** (math.ceil(math.log(1.0 / (factor - 1.0))) + 1)
    checkpoints = []
    while current < threshold:
        checkpoints.append(round(current))
        current += 1
    while round(current) < end:
        checkpoints.append(round(current))
        current *= factor
    checkpoints.append(round(end))
    return checkpoints


def _evaluation_steps(base_step: int, total_steps: int, frequency: int) -> set[int]:
    factor = 2 ** (1.0 / frequency)
    multiplier = 2 ** (math.ceil(math.log(1.0 / (factor - 1.0))) + 1)
    log_steps = _log2_checkpoints(multiplier * base_step, frequency)
    current = multiplier * factor
    while round(current) * base_step < total_steps:
        log_steps.append(round(current) * base_step)
        current *= factor
    log_steps.append(total_steps)
    return {int(value) for value in log_steps if 0 < value <= total_steps}


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: CausalTransformer,
    data_config: DataConfig,
    rules: Rules,
    seeds: dict[str, int],
    epoch: int,
    train_config: dict[str, Any],
    *,
    best_model_state_dict: dict[str, torch.Tensor] | None = None,
    best_step: int | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "data_config": data_config.to_dict(),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "best_model_state_dict": best_model_state_dict or model.state_dict(),
        "last_model_state_dict": model.state_dict(),
        "best_step": int(best_step if best_step is not None else epoch),
        "rules": rules.to_dict(),
        "seeds": {key: int(value) for key, value in seeds.items()},
        "epoch": int(epoch),
        "train_config": dict(train_config),
    }
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[CausalTransformer, DataConfig, Rules, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    data_config = DataConfig.from_dict(payload["data_config"])
    model_config = TransformerConfig.from_dict(payload["model_config"])
    rules = Rules.from_dict(payload["rules"])
    model = CausalTransformer(model_config)
    model.load_state_dict(payload.get("best_model_state_dict", payload["model_state_dict"]))
    model.to(device)
    model.eval()
    return model, data_config, rules, payload


def _evaluate(
    model: CausalTransformer,
    tokens: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = None
    correct_sum = None
    examples = 0
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start : start + batch_size], device=device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            losses = nn.functional.cross_entropy(
                logits.transpose(1, 2), targets, reduction="none"
            )
            correct = logits.argmax(dim=-1).eq(targets)
            batch_loss = losses.sum(dim=0).cpu().numpy()
            batch_correct = correct.sum(dim=0).cpu().numpy()
            loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
            correct_sum = batch_correct if correct_sum is None else correct_sum + batch_correct
            examples += len(batch)
    return loss_sum / examples, correct_sum / examples


def train_from_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Train one model and write checkpoints plus a CSV learning curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = DataConfig.from_dict(config["data"])
    train_config = dict(config["training"])
    seeds = {key: int(value) for key, value in config.get("seeds", {}).items()}
    if "random_seed" in config:
        random.seed(int(config["random_seed"]))
        seeds.update(
            random_seed=int(config["random_seed"]),
            rules=random.randint(10_000_000, 99_999_999),
            train_data=random.randint(10_000_000, 99_999_999),
            model=random.randint(10_000_000, 99_999_999),
        )
    device = resolve_device(train_config.get("device", "auto"))

    random.seed(seeds["model"])
    np.random.seed(seeds["model"])
    torch.manual_seed(seeds["model"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["model"])

    rules = build_rules(data_config, seeds["rules"])
    train_count = int(train_config["train_examples"])
    validation_count = int(train_config["validation_examples"])
    corpus = sample_corpus(
        train_count + validation_count, data_config, rules, seeds["train_data"]
    )
    train_tokens = corpus["tokens"][:train_count]
    validation_tokens = corpus["tokens"][train_count:]

    model_values = dict(config["model"])
    model_config = TransformerConfig(
        vocab_size=data_config.surface_vocab_size,
        context_length=data_config.sequence_length,
        **model_values,
    )
    model = CausalTransformer(model_config).to(device)
    tokens = torch.as_tensor(train_tokens, dtype=torch.long)
    if int(train_config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("the paper checkpoints use gradient_accumulation_steps=1")
    loader = DataLoader(
        TensorDataset(tokens),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(float(train_config.get("weight_decay", 0.0))),
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config.get("beta1", 0.9)), float(train_config.get("beta2", 0.999))),
        eps=1e-8,
    )
    epochs = int(train_config["epochs"])
    total_steps = epochs * math.ceil(train_count / int(train_config["batch_size"]))
    scheduler_name = str(train_config.get("scheduler", "wsd"))
    if scheduler_name != "wsd":
        raise ValueError("the paper configuration uses the wsd scheduler")
    scheduler = WSDLR(
        optimizer,
        int(float(train_config.get("warmup_fraction", 0.02)) * total_steps),
        int(float(train_config.get("decay_fraction", 0.1)) * total_steps),
        total_steps,
        float(train_config.get("min_lr_factor", 0.0)),
    )

    history = []
    evaluation_steps = _evaluation_steps(
        int(train_config.get("print_frequency", 2048)),
        total_steps,
        int(train_config.get("save_frequency", 4)),
    )
    best_last_token_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    step = 0
    stop = False
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_targets = 0
        for (batch,) in loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            train_loss_sum += float(loss.detach()) * targets.numel()
            train_targets += targets.numel()
            if step in evaluation_steps:
                validation_loss, validation_accuracy = _evaluate(
                    model,
                    validation_tokens,
                    int(train_config.get("evaluation_batch_size", 1024)),
                    device,
                )
                row = {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": train_loss_sum / train_targets,
                    "validation_loss": float(validation_loss.mean()),
                    "last_token_validation_loss": float(validation_loss[-1]),
                    "last_token_validation_accuracy": float(validation_accuracy[-1]),
                }
                if row["last_token_validation_loss"] < best_last_token_loss:
                    best_last_token_loss = row["last_token_validation_loss"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_step = step
                history.append(row)
                print(
                    f"step {step:06d}/{total_steps}: train_loss={row['train_loss']:.6f} "
                    f"validation_loss={row['validation_loss']:.6f} "
                    f"last_token_loss={row['last_token_validation_loss']:.6f}"
                )
        epoch_loss = train_loss_sum / train_targets
        if epoch_loss <= float(train_config.get("loss_threshold", 1e-6)):
            if not history or history[-1]["step"] != step:
                validation_loss, validation_accuracy = _evaluate(
                    model,
                    validation_tokens,
                    int(train_config.get("evaluation_batch_size", 1024)),
                    device,
                )
                row = {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": epoch_loss,
                    "validation_loss": float(validation_loss.mean()),
                    "last_token_validation_loss": float(validation_loss[-1]),
                    "last_token_validation_accuracy": float(validation_accuracy[-1]),
                }
                if row["last_token_validation_loss"] < best_last_token_loss:
                    best_last_token_loss = row["last_token_validation_loss"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_step = step
                history.append(row)
            stop = True
        if stop:
            break

    final_path = output_dir / "checkpoint.pt"
    save_checkpoint(
        final_path,
        model,
        data_config,
        rules,
        seeds,
        epochs,
        train_config,
        best_model_state_dict=best_state,
        best_step=best_step,
    )
    with (output_dir / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return final_path
