from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    level: float = 0.95

    def width(self) -> float:
        return self.upper - self.lower

    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class Estimate:
    name: str
    value: float
    interval: Optional[Interval] = None
    p_value: Optional[float] = None
    unit: str = ""

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value is not None and self.p_value < alpha


@dataclass(frozen=True)
class CohortRecord:
    cohort_id: int
    group: str
    sample_size: int
    adr_percent: float
    fidelity_score: Optional[float]
    volume_tier: str
    annual_volume: Optional[float] = None
    baseline_adr: Optional[float] = None
    mean_age: Optional[float] = None
    female_fraction: Optional[float] = None
    screening_fraction: Optional[float] = None
    adequate_prep_fraction: Optional[float] = None

    def events(self) -> int:
        return int(round(self.sample_size * self.adr_percent / 100.0))

    def non_events(self) -> int:
        return self.sample_size - self.events()

    def is_supported(self) -> bool:
        return self.group == "implementation"


@dataclass(frozen=True)
class OutcomeRecord:
    name: str
    implementation_value: float
    unsupported_value: float
    difference: float
    adjusted_odds_ratio: Optional[float]
    confidence_interval: Optional[Interval]


@dataclass(frozen=True)
class FidelityComponents:
    training: float
    feedback: float
    alert_management: float

    def total(self) -> float:
        return self.training + self.feedback + self.alert_management

    def normalized(self) -> Tuple[float, float, float]:
        scale = 100.0 / 33.3
        return self.training * scale, self.feedback * scale, self.alert_management * scale


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 2025
    confidence: float = 0.95
    alpha: float = 0.05
    secondary_alpha: float = 0.007
    rct_benchmark_pp: float = 8.1
    meaningful_gain_pp: float = 5.0
    spline_knots: Tuple[float, float, float] = (0.25, 0.5, 0.75)
    fidelity_cutoff: float = 75.0
    sensitivity_cutoff: float = 80.0
    bootstrap_repetitions: int = 200000


@dataclass(frozen=True)
class ComputeConfig:
    world_size: int
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_width: int
    hidden_depth: int
    ensemble_members: int

    def effective_batch(self) -> int:
        return self.world_size * self.batch_size * self.gradient_accumulation


@dataclass
class AnalysisBundle:
    estimates: List[Estimate] = field(default_factory=list)
    tables: Dict[str, List[Mapping[str, object]]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_estimate(self, estimate: Estimate) -> None:
        self.estimates.append(estimate)

    def add_table(self, name: str, records: List[Mapping[str, object]]) -> None:
        self.tables[name] = records

    def set_metadata(self, key: str, value: object) -> None:
        self.metadata[key] = value


@dataclass(frozen=True)
class PathLayout:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> Path:
        return self.root / "configs" / "primary.yaml"

    @property
    def output(self) -> Path:
        return self.root / "outputs"


Numeric = Sequence[float]
Matrix = Sequence[Sequence[float]]
NamedValues = Mapping[str, float]
