import math
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class NetworkShape:
    input_features: int = 96
    hidden_width: int = 8192
    hidden_depth: int = 48
    attention_heads: int = 64
    dropout: float = 0.05
    outcomes: int = 7

    def validate(self) -> None:
        if self.input_features <= 0:
            raise ValueError("input_features must be positive")
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if self.hidden_depth <= 0:
            raise ValueError("hidden_depth must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_width % self.attention_heads != 0:
            raise ValueError("hidden_width must divide by attention_heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout outside valid range")


@dataclass(frozen=True)
class NetworkOutput:
    logits: Tensor
    mean: Tensor
    log_variance: Tensor
    fidelity: Tensor
    representation: Tensor


class FeatureAffine(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(features, dtype=torch.float64))

    def forward(self, values: Tensor) -> Tensor:
        return values * self.weight + self.bias


class StableLayerNorm(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.features = features
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(features, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(features, dtype=torch.float64))

    def forward(self, values: Tensor) -> Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        centered = values - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias


class GatedActivation(nn.Module):
    def __init__(self, features: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = features * expansion
        self.value = nn.Linear(features, hidden, dtype=torch.float64)
        self.gate = nn.Linear(features, hidden, dtype=torch.float64)
        self.output = nn.Linear(hidden, features, dtype=torch.float64)

    def forward(self, values: Tensor) -> Tensor:
        activated = functional.silu(self.gate(values)) * self.value(values)
        return self.output(activated)


class ResidualGate(nn.Module):
    def __init__(self, features: int, initial: float = 0.01) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((features,), initial, dtype=torch.float64))

    def forward(self, residual: Tensor, update: Tensor) -> Tensor:
        return residual + update * self.scale


class FeatureAttention(nn.Module):
    def __init__(self, features: int, heads: int, dropout: float) -> None:
        super().__init__()
        if features % heads != 0:
            raise ValueError("features must divide by heads")
        self.features = features
        self.heads = heads
        self.head_width = features // heads
        self.scale = self.head_width**-0.5
        self.query = nn.Linear(features, features, bias=False, dtype=torch.float64)
        self.key = nn.Linear(features, features, bias=False, dtype=torch.float64)
        self.value = nn.Linear(features, features, bias=False, dtype=torch.float64)
        self.output = nn.Linear(features, features, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)

    def split(self, values: Tensor) -> Tensor:
        batch, tokens, _ = values.shape
        return values.reshape(batch, tokens, self.heads, self.head_width).transpose(1, 2)

    def merge(self, values: Tensor) -> Tensor:
        batch, _, tokens, _ = values.shape
        return values.transpose(1, 2).contiguous().reshape(batch, tokens, self.features)

    def forward(self, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query = self.split(self.query(values))
        key = self.split(self.key(values))
        projected = self.split(self.value(values))
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if mask is not None:
            expanded = mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~expanded, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, projected)
        return self.output(self.merge(attended))


class CohortBlock(nn.Module):
    def __init__(self, features: int, heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = StableLayerNorm(features)
        self.attention = FeatureAttention(features, heads, dropout)
        self.attention_gate = ResidualGate(features)
        self.feedforward_norm = StableLayerNorm(features)
        self.feedforward = GatedActivation(features, expansion)
        self.feedforward_gate = ResidualGate(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        attention_update = self.attention(self.attention_norm(values), mask)
        values = self.attention_gate(values, self.dropout(attention_update))
        feedforward_update = self.feedforward(self.feedforward_norm(values))
        return self.feedforward_gate(values, self.dropout(feedforward_update))


class ContinuousTokenizer(nn.Module):
    def __init__(self, features: int, width: int) -> None:
        super().__init__()
        self.features = features
        self.width = width
        self.weight = nn.Parameter(torch.empty(features, width, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(features, width, dtype=torch.float64))
        self.missing = nn.Parameter(torch.empty(features, width, dtype=torch.float64))
        nn.init.normal_(self.weight, std=width**-0.5)
        nn.init.normal_(self.missing, std=width**-0.5)

    def forward(self, values: Tensor, observed: Optional[Tensor] = None) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.features:
            raise ValueError("feature matrix has unexpected shape")
        tokens = values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        if observed is None:
            return tokens
        missing_tokens = self.missing.unsqueeze(0).expand(values.shape[0], -1, -1)
        return torch.where(observed.unsqueeze(-1).to(dtype=torch.bool), tokens, missing_tokens)


class CategoricalTokenizer(nn.Module):
    def __init__(self, cardinalities: Sequence[int], width: int) -> None:
        super().__init__()
        self.cardinalities = tuple(cardinalities)
        offsets: List[int] = []
        running = 0
        for cardinality in cardinalities:
            if cardinality <= 0:
                raise ValueError("cardinality must be positive")
            offsets.append(running)
            running += cardinality + 1
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=True)
        self.embedding = nn.Embedding(running, width, dtype=torch.float64)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != len(self.cardinalities):
            raise ValueError("categorical matrix has unexpected shape")
        shifted = values.to(dtype=torch.long) + self.offsets.unsqueeze(0)
        return self.embedding(shifted)


class CohortEncoder(nn.Module):
    def __init__(self, shape: NetworkShape, categorical_cardinalities: Sequence[int] = ()) -> None:
        super().__init__()
        shape.validate()
        self.shape = shape
        self.continuous = ContinuousTokenizer(shape.input_features, shape.hidden_width)
        self.categorical = CategoricalTokenizer(categorical_cardinalities, shape.hidden_width) if categorical_cardinalities else None
        token_count = shape.input_features + len(categorical_cardinalities) + 1
        self.summary = nn.Parameter(torch.empty(1, 1, shape.hidden_width, dtype=torch.float64))
        self.position = nn.Parameter(torch.empty(1, token_count, shape.hidden_width, dtype=torch.float64))
        self.blocks = nn.ModuleList(
            [CohortBlock(shape.hidden_width, shape.attention_heads, shape.dropout) for _ in range(shape.hidden_depth)]
        )
        self.norm = StableLayerNorm(shape.hidden_width)
        nn.init.normal_(self.summary, std=shape.hidden_width**-0.5)
        nn.init.normal_(self.position, std=shape.hidden_width**-0.5)

    def forward(self, continuous: Tensor, observed: Optional[Tensor] = None, categorical: Optional[Tensor] = None) -> Tensor:
        tokens = self.continuous(continuous, observed)
        if self.categorical is not None:
            if categorical is None:
                raise ValueError("categorical values required")
            tokens = torch.cat([tokens, self.categorical(categorical)], dim=1)
        summary = self.summary.expand(continuous.shape[0], -1, -1)
        values = torch.cat([summary, tokens], dim=1)
        values = values + self.position[:, : values.shape[1]]
        for block in self.blocks:
            values = block(values)
        return self.norm(values[:, 0])


class HeteroscedasticHead(nn.Module):
    def __init__(self, width: int, outcomes: int) -> None:
        super().__init__()
        self.mean = nn.Linear(width, outcomes, dtype=torch.float64)
        self.log_variance = nn.Linear(width, outcomes, dtype=torch.float64)

    def forward(self, values: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.mean(values)
        log_variance = torch.clamp(self.log_variance(values), min=-12.0, max=8.0)
        return mean, log_variance


class FidelityHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width // 2, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(width // 2, width // 8, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(width // 8, 1, dtype=torch.float64),
        )

    def forward(self, values: Tensor) -> Tensor:
        return 100.0 * torch.sigmoid(self.layers(values)).squeeze(-1)


class MultiOutcomeModel(nn.Module):
    def __init__(self, shape: NetworkShape, categorical_cardinalities: Sequence[int] = ()) -> None:
        super().__init__()
        self.encoder = CohortEncoder(shape, categorical_cardinalities)
        self.binary_head = nn.Linear(shape.hidden_width, shape.outcomes, dtype=torch.float64)
        self.continuous_head = HeteroscedasticHead(shape.hidden_width, shape.outcomes)
        self.fidelity_head = FidelityHead(shape.hidden_width)

    def forward(self, continuous: Tensor, observed: Optional[Tensor] = None, categorical: Optional[Tensor] = None) -> NetworkOutput:
        representation = self.encoder(continuous, observed, categorical)
        logits = self.binary_head(representation)
        mean, log_variance = self.continuous_head(representation)
        fidelity = self.fidelity_head(representation)
        return NetworkOutput(logits, mean, log_variance, fidelity, representation)


class EnsembleModel(nn.Module):
    def __init__(self, members: Sequence[MultiOutcomeModel]) -> None:
        super().__init__()
        if len(members) == 0:
            raise ValueError("ensemble requires members")
        self.members = nn.ModuleList(members)

    def forward(self, continuous: Tensor, observed: Optional[Tensor] = None, categorical: Optional[Tensor] = None) -> Sequence[NetworkOutput]:
        return [member(continuous, observed, categorical) for member in self.members]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def gaussian_negative_log_likelihood(mean: Tensor, log_variance: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    inverse_variance = torch.exp(-log_variance)
    values = 0.5 * (log_variance + (target - mean).square() * inverse_variance)
    if mask is not None:
        values = values * mask.to(dtype=values.dtype)
        denominator = torch.clamp(mask.sum(), min=1).to(dtype=values.dtype)
        return values.sum() / denominator
    return values.mean()


def focal_binary_loss(logits: Tensor, target: Tensor, gamma: float = 2.0, alpha: float = 0.25, mask: Optional[Tensor] = None) -> Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    target_probability = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_weight = alpha * target + (1.0 - alpha) * (1.0 - target)
    values = alpha_weight * (1.0 - target_probability).pow(gamma) * cross_entropy
    if mask is not None:
        values = values * mask.to(dtype=values.dtype)
        return values.sum() / torch.clamp(mask.sum(), min=1).to(dtype=values.dtype)
    return values.mean()


def fidelity_loss(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    values = functional.smooth_l1_loss(prediction, target, reduction="none", beta=5.0)
    values = values * observed.to(dtype=values.dtype)
    return values.sum() / torch.clamp(observed.sum(), min=1).to(dtype=values.dtype)


def pairwise_ranking_loss(prediction: Tensor, target: Tensor, temperature: float = 1.0) -> Tensor:
    prediction_difference = prediction.unsqueeze(1) - prediction.unsqueeze(0)
    target_difference = target.unsqueeze(1) - target.unsqueeze(0)
    direction = torch.sign(target_difference)
    active = direction != 0.0
    values = functional.softplus(-direction * prediction_difference / temperature)
    return values[active].mean() if torch.any(active) else values.sum() * 0.0


def representation_penalty(representation: Tensor) -> Tensor:
    centered = representation - representation.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, representation.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance - torch.diag_embed(diagonal)
    variance_penalty = torch.mean((diagonal - 1.0).square())
    covariance_penalty = off_diagonal.square().mean()
    return variance_penalty + covariance_penalty


def total_training_loss(output: NetworkOutput, binary_target: Tensor, continuous_target: Tensor, fidelity_target: Tensor, binary_mask: Tensor, continuous_mask: Tensor, fidelity_observed: Tensor) -> Mapping[str, Tensor]:
    binary = focal_binary_loss(output.logits, binary_target, mask=binary_mask)
    continuous = gaussian_negative_log_likelihood(output.mean, output.log_variance, continuous_target, continuous_mask)
    fidelity = fidelity_loss(output.fidelity, fidelity_target, fidelity_observed)
    ranking = pairwise_ranking_loss(output.fidelity, fidelity_target)
    representation = representation_penalty(output.representation)
    total = binary + continuous + 0.1 * fidelity + 0.05 * ranking + 0.001 * representation
    return {
        "total": total,
        "binary": binary,
        "continuous": continuous,
        "fidelity": fidelity,
        "ranking": ranking,
        "representation": representation,
    }


def initialize_model(model: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            standard_deviation = math.sqrt(2.0 / fan_in)
            values = torch.randn(module.weight.shape, generator=generator, dtype=module.weight.dtype) * standard_deviation
            with torch.no_grad():
                module.weight.copy_(values)
                if module.bias is not None:
                    module.bias.zero_()


def parameter_groups(model: nn.Module, weight_decay: float) -> List[Mapping[str, object]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name or "position" in name or "summary" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def gradient_norm(model: nn.Module) -> Tensor:
    norms = [parameter.grad.detach().norm(2) for parameter in model.parameters() if parameter.grad is not None]
    if not norms:
        return torch.zeros((), dtype=torch.float64)
    return torch.stack(norms).norm(2)


def model_parameter_count(shape: NetworkShape) -> int:
    model = MultiOutcomeModel(shape)
    return sum(parameter.numel() for parameter in model.parameters())
